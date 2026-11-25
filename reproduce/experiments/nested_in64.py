#!/usr/bin/env python
"""Guiding with an earlier checkpoint against guiding with a retrained model (EDM2-XS, ImageNet-64; fig:teaser-nested,
App. A.5 and B.7).

Two EDM2-XS runs that differ only in their seed, A (seed 0) and B (seed 1), each T = 2^27 training images with an
EMA snapshot every 2^20 images; a budget k counts those units (T = 128). The strong model is A or B at s = T or
T/2; its weak model is the snapshot at k = r s, r in {1/8, 1/4, 1/2, 3/4}, from the same run (a checkpoint,
"nested") or from the other run (an independent model, "indep"). Every model is the EMA of std 0.100 that EDM2
tracks during training (the raw snapshot, no post-hoc reconstruction).

    python reproduce/experiments/seedpair_in64.py prep                  # CPU: the ImageNet-64 training zip
    sbatch ... --gres=gpu:8 reproduce/slurm/train_xs_2e27.sbatch A 0    # GPU: run A (then B 1)
    python reproduce/experiments/nested_in64.py gram --shard {task} --shards 4   # GPU: the Gram of the snapshots
    python reproduce/experiments/nested_in64.py sweep --task {task} --tasks 8    # GPU: 292 arms, FID / FD_DINOv2
    python reproduce/experiments/nested_in64.py analyze    # CPU: w* per pair and metric -> summary.json
    python reproduce/experiments/nested_in64.py fit        # CPU: the moving-average model's L -> fit.json
    python reproduce/experiments/nested_in64.py pairs      # the 32 (strong, weak) pairs
    python reproduce/experiments/nested_in64.py test       # CPU: the fits on synthetic data
    gram / sweep --smoke     # NVIDIA's XS snapshots stand in for ours, a few images (one GPU)

gram: predictions of the 22 snapshots (both runs, BUDGETS) on 4,096 training images with their labels, at the 32
levels of EDM2's Heun sampler, one noise draw per image and level shared by every snapshot. Per level it keeps
G[k, l] = sum_i <D_k - xbar, D_l - xbar>, H[k] = sum_i <D_k - xbar, x0 - xbar> and X2 = sum_i |x0 - xbar|^2.
sweep: D_w = D_weak + w (D_strong - D_weak) at w in WS, sampled and scored by ens_edm2.py: 10,000 samples, seeds
0..9,999 (the class from the seed), EDM2's deterministic Heun sampler with 32 steps, FID and FD_DINOv2 against
NVIDIA's ImageNet-64 training statistics. w = 1 is the strong model alone, one arm per strong model. The samples
and features of a scored arm are deleted unless --keep.
analyze: the optimal w is the minimum of a quadratic in log w through the five points around the measured minimum
(the minimum itself when it lies at an end of WS); a ratio is w*_nested / w*_indep for one strong model and r.
fit: the moving-average model of App. A.5, D_k = sum_i c_{k,i} x0^(i) with c_{k,i} proportional to lambda^(k-i),
one step per 2^20 images, L = 1 / (1 - lambda). Its ratio is sqrt((V_s + V_k) / (V_s + V_k - 2 V_sk)); L is fitted
by least squares on a grid to the eight targets (two metrics x four r, each the mean of four ratios: two seeds, two
strong budgets), with the strong model at T for every target. From the Gram: the seed-noise correlation of a
checkpoint with the final model, Cov(A_T, A_k) / Var(A_T), against V_Tk / V_T, and the spread ratio
Var(A_k) / Var(A_T) against V_k / V_T, each pooled over levels, give two more estimates of L.

Outputs under $VMG_RUNS/edm2-64-xs-2e27/: train/<run>/ (train_xs_2e27.sbatch), gram/level<j>.npz, sweep/ (ens_edm2.py
--out: samples/, metrics/<arm>.json), summary.json, fit.json; --smoke writes under smoke/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

UNIT = 2 ** 20                                  # images per snapshot step
T = 128                                         # the full budget, 2^27 images
RUNS = {"A": 0, "B": 1}                         # run -> seed
BUDGETS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
EMA = "0.100"
STRONG = (T, T // 2)
RS = (1 / 8, 1 / 4, 1 / 2, 3 / 4)
WS = (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6.5, 8)
METRICS = ("fid", "fd_dinov2")
NEVAL = 10000
M_GRAM, GRAM_SEED = 4096, 123
L_GRID = np.linspace(2, 400, 3981)              # L in snapshot steps, 0.1 apart
SMOKE_NETS = ("edm2-img64-xs-0033554-0.100", "edm2-img64-xs-0033554-0.050")   # NVIDIA's, for --smoke


def root(smoke=False):
    from vmg import runs
    d = runs.root() / "edm2-64-xs-2e27"
    return d / "smoke" if smoke else d


def snapshot(run, k, ema=EMA):
    """Run `run`'s EMA snapshot after k x 2^20 training images."""
    return root() / "train" / run / f"network-snapshot-{k * UNIT // 1000:07d}-{ema}.pkl"


def model(run, k):
    return f"{run}{k}"


def pairs():
    """[(strong, weak, kind, s, k)]: strong run at s, the weak snapshot at k = r s from the same or the other run."""
    out = []
    for s in STRONG:
        for S in RUNS:
            O = next(r for r in RUNS if r != S)
            for r in RS:
                k = round(r * s)
                out += [(model(S, s), model(S, k), "nested", s, k), (model(S, s), model(O, k), "indep", s, k)]
    return out


def arms():
    """[(arm, d1, d0, w)] of the sweep: each strong model alone (w = 1), then every pair at the other weights."""
    strong = list(dict.fromkeys(p[0] for p in pairs()))
    out = [(f"{st}-w1", st, "", 1.0) for st in strong]
    return out + [(f"{st}-{wk}-w{w:g}", st, wk, w) for st, wk, *_ in pairs() for w in WS if w != 1]


def checkpoint_path(name):
    """A sweep member (e.g. A128) -> its snapshot."""
    return snapshot(name[0], int(name[1:]))


# ------------------------------------------------------------------ GPU: the Gram of the snapshots' predictions
def gram(a):
    import pickle
    import torch
    from reproduce.experiments import seedpair_in64 as SP
    from vmg.models.edm2 import EDMHeun
    SP.edm2_on_path()
    out = root(a.smoke) / "gram"
    out.mkdir(parents=True, exist_ok=True)
    sig = EDMHeun().levels()["sigma"]
    if a.smoke:
        from vmg.models import _edm
        paths = {f"{r}{T}@{EMA}": _edm.fetch(_edm.raw(n)) for r, n in zip(RUNS, SMOKE_NETS)}
        levels, m = [0, len(sig) - 1], 16
    else:
        paths = {f"{model(r, k)}@{EMA}": snapshot(r, k) for r in RUNS for k in BUDGETS}
        levels, m = [j for j in range(len(sig)) if j % a.shards == a.shard], a.m
    todo = [j for j in levels if not (out / f"level{j:02d}.npz").exists()]
    if not todo:
        print("every level is done")
        return
    idx = np.random.default_rng(GRAM_SEED).choice(SP.M_TRAIN, m, replace=False)
    u8, lab = SP.zip_images(idx)
    x0 = torch.from_numpy(u8).cuda().float() / 127.5 - 1
    onehot = torch.nn.functional.one_hot(torch.from_numpy(lab).cuda(), 1000).float()
    xbar = x0.mean(0, keepdim=True)
    names = list(paths)
    nets = [pickle.load(open(paths[n], "rb"))["ema"].cuda().eval().requires_grad_(False) for n in names]
    K = len(names)
    for j in todo:
        s = float(sig[j])
        eps = torch.randn(x0.shape, generator=torch.Generator(device="cuda").manual_seed(1000 + j), device="cuda")
        G, H, X2 = np.zeros((K, K)), np.zeros(K), 0.0
        for b in range(0, m, a.batch):
            xb, lb = x0[b:b + a.batch], onehot[b:b + a.batch]
            xt, sg = xb + s * eps[b:b + a.batch], torch.full((len(xb),), s, device="cuda")
            with torch.no_grad():
                P = torch.stack([net(xt, sg, lb).float() for net in nets])
            P = (P - xbar).flatten(2).double()                                   # [K, B, d]
            R = (xb - xbar).flatten(1).double()                                  # [B, d]
            G += torch.einsum("kbd,lbd->kl", P, P).cpu().numpy()
            H += torch.einsum("kbd,bd->k", P, R).cpu().numpy()
            X2 += float((R ** 2).sum())
        tmp = out / f".level{j:02d}.{os.getpid()}.npz"
        np.savez(tmp, G=G, H=H, X2=X2, M=m, sigma=s, names=np.array(names), idx=idx)
        os.replace(tmp, out / f"level{j:02d}.npz")
        print(f"level {j:2d} sigma {s:8.4f} done", flush=True)


# ------------------------------------------------------------------ GPU: the guidance sweeps (ens_edm2.py)
def ens(*args):
    """Run ens_edm2.py's command line in this process (its networks and detectors stay loaded across arms)."""
    from reproduce.experiments import ens_edm2
    argv = sys.argv
    sys.argv = ["ens_edm2.py", *map(str, args)]
    try:
        ens_edm2.main()
    finally:
        sys.argv = argv


def sweep(a):
    import shutil
    out = root(a.smoke) / "sweep"
    if a.smoke:
        from vmg.models import _edm
        todo, n = [("smoke-w1", "S", "", 1.0), ("smoke-w2", "S", "W", 2.0)], 8
        ckpt = {"S": _edm.fetch(_edm.raw(SMOKE_NETS[0])), "W": _edm.fetch(_edm.raw(SMOKE_NETS[1]))}
    else:
        todo, n = arms()[a.task::a.tasks], NEVAL
        ckpt = {m: checkpoint_path(m) for _, d1, d0, _ in todo for m in (d1, d0) if m}
    for arm, d1, d0, w in todo:
        if (out / "metrics" / f"{arm}.json").exists():
            continue
        members = [f"--ckpt={m}={ckpt[m]}" for m in (d1, d0) if m]
        ens("sample", "--arm", arm, "--d1", d1, f"--d0={d0}", f"--w={w:g}", "--out", out, "--neval", n, *members)
        ens("eval", "--arm", arm, "--n", n, "--out", out, *members)
        if not a.keep:
            shutil.rmtree(out / "samples" / arm)
            (out / "metrics" / f"{arm}.feats.npz").unlink(missing_ok=True)


# ------------------------------------------------------------------ CPU: optimal weights and their ratios
def optimum(ws, vals):
    """(w*, metric at w*, at_edge): the vertex of a quadratic in log w through the five points around the minimum,
    clipped to their range; the minimum itself if it lies at an end of the grid (or the fit is not convex)."""
    ws, vals = np.asarray(ws, float), np.asarray(vals, float)
    j = int(vals.argmin())
    edge = j in (0, len(ws) - 1)
    lo, hi = max(0, j - 2), min(len(ws), j + 3)
    x, y = np.log(ws[lo:hi]), vals[lo:hi]
    if not edge:
        p = np.polyfit(x, y, 2)
        if p[0] > 0:
            xo = float(np.clip(-p[1] / (2 * p[0]), x.min(), x.max()))
            return float(np.exp(xo)), float(np.polyval(p, xo)), False
    return float(ws[j]), float(vals[j]), edge


def analyze(a):
    d = root() / "sweep" / "metrics"
    rec = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}
    missing = [arm for arm, *_ in arms() if arm not in rec]
    assert not missing, f"{len(missing)} arms not scored in {d}, e.g. {missing[:3]} (nested_in64.py sweep)"
    opt = {m: [] for m in METRICS}
    for st, wk, kind, s, k in pairs():
        curve = [rec[f"{st}-w1"]] + [rec[f"{st}-{wk}-w{w:g}"] for w in WS if w != 1]
        assert all(r["n"] == NEVAL for r in curve), f"{st} <- {wk}: not every arm has {NEVAL} samples"
        for m in METRICS:
            vals = [r[m] for r in curve]
            w, v, edge = optimum(WS, vals)
            opt[m].append(dict(strong=st, weak=wk, kind=kind, s=s, k=k, wopt=w, vmin=v, at_edge=edge, vals=vals))
    summary = dict(T=T, unit=UNIT, ema=EMA, ws=list(WS), opt=opt)
    (root() / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{'metric':10s} {'strong':>6s} {'r':>5s} {'w* nested':>10s} {'w* indep':>9s} {'ratio':>6s}")
    for m in METRICS:
        for s in STRONG:
            for S in RUNS:
                for r in RS:
                    n, i = (pick(summary, m, s, S, r, kind)["wopt"] for kind in ("nested", "indep"))
                    print(f"{m:10s} {S}{s:>5d} {r:5.3f} {n:10.3f} {i:9.3f} {n / i:6.3f}")
    print(f"wrote {root() / 'summary.json'}")


def pick(summary, m, s, S, r, kind):
    hit = [o for o in summary["opt"][m] if o["s"] == s and o["k"] == round(r * s) and o["kind"] == kind
           and o["strong"].startswith(S)]
    assert len(hit) == 1, (m, s, S, r, kind)
    return hit[0]


def ratio(summary, m, s, S, r):
    """w*_nested / w*_indep of metric m for strong run S at budget s and fraction r."""
    return pick(summary, m, s, S, r, "nested")["wopt"] / pick(summary, m, s, S, r, "indep")["wopt"]


def ratios(summary, m, r):
    """The four measured ratios at fraction r: strong budget T then T/2, strong run A then B."""
    return np.array([ratio(summary, m, s, S, r) for s in STRONG for S in RUNS])


# ------------------------------------------------------------------ CPU: the moving-average model
def ema_weights(k, lam):
    c = lam ** np.arange(k - 1, -1, -1.0)
    return c / c.sum()


def ema_terms(s, k, lam):
    """(V_s, V_k, V_sk) of the moving average over one target per step: the strong model at step s, its
    checkpoint at step k <= s."""
    cs, ck = ema_weights(s, lam), ema_weights(k, lam)
    return (cs ** 2).sum(), (ck ** 2).sum(), (cs[:k] * ck).sum()


def ema_ratio(s, k, lam):
    """w*_checkpoint / w*_independent of the moving-average model (App. A.5, eq. ratio-formula-full)."""
    vs, vk, vsk = ema_terms(s, k, lam)
    return np.sqrt((vs + vk) / (vs + vk - 2 * vsk))


def fit_L(points, target, f):
    """The grid L (in steps) minimising sum (f(s, k, lambda) - target)^2 over points [(s, k)]; returns (L, sse)."""
    target = np.asarray(target, float)
    sse = [float(((np.array([f(s, k, np.exp(-1 / L)) for s, k in points]) - target) ** 2).sum()) for L in L_GRID]
    i = int(np.argmin(sse))
    return float(L_GRID[i]), sse[i]


def load_gram(d):
    files = sorted(Path(d).glob("level*.npz"))
    assert files, f"no level*.npz in {d} (nested_in64.py gram)"
    zs = [dict(np.load(f)) for f in files]
    names = [str(n) for n in zs[0]["names"]]
    return names, np.stack([z["G"] / float(z["M"]) for z in zs]), np.array([float(z["sigma"]) for z in zs])


def seed_noise(names, G, s, k, ema=EMA):
    """Pooled over levels: (Var of one run at s, of one run at k, Cov of a run at s with its own snapshot at k),
    each per image, from E|D_a - D_b|^2 between the four snapshots (the two mirrors averaged)."""
    ix = {n: i for i, n in enumerate(names)}
    nm = lambda r, b: ix[f"{model(r, b)}@{ema}"]                               # noqa: E731
    d2 = lambda i, j: G[:, i, i] + G[:, j, j] - 2 * G[:, i, j]                 # noqa: E731
    spread = lambda b: d2(nm("A", b), nm("B", b)) / 2                          # noqa: E731
    cross = np.mean([(d2(nm(S, s), nm(O, k)) - d2(nm(S, s), nm(S, k))) / 2 for S, O in (("A", "B"), ("B", "A"))], 0)
    return float(spread(s).sum()), float(spread(k).sum()), float(cross.sum())


def diagnostics(names, G, s=T):
    """L from the seed noise of the unguided snapshots (strong at s): its correlation with the final model's, and
    its spread relative to the final model's."""
    pts = [(s, round(r * s)) for r in RS]
    sn = [seed_noise(names, G, a, b) for a, b in pts]
    rho = [c / vs for vs, _, c in sn]
    spread = [vk / vs for vs, vk, _ in sn]
    L_rho, _ = fit_L(pts, rho, lambda a, b, lam: ema_terms(a, b, lam)[2] / ema_terms(a, b, lam)[0])
    L_spread, _ = fit_L(pts, spread, lambda a, b, lam: ema_terms(a, b, lam)[1] / ema_terms(a, b, lam)[0])
    return dict(r=list(RS), memory=rho, spread_ratio=spread, L_memory=L_rho / T, L_spread=L_spread / T)


def fit(a):
    summary = json.loads((root() / "summary.json").read_text())
    q = {m: [ratios(summary, m, r) for r in RS] for m in METRICS}
    target = [x.mean() for m in METRICS for x in q[m]]
    pts = [(T, round(r * T)) for r in RS] * len(METRICS)
    L, sse = fit_L(pts, target, ema_ratio)
    lam = np.exp(-1 / L)
    res = dict(L=L / T, L_steps=L, sse=sse, r=list(RS), targets={m: [float(x.mean()) for x in q[m]] for m in METRICS},
               range={m: [[float(x.min()), float(x.max())] for x in q[m]] for m in METRICS},
               model=[float(ema_ratio(T, round(r * T), lam)) for r in RS],
               no_forgetting=[float(np.sqrt((1 + r) / (1 - r))) for r in RS])
    own = [(s, round(r * s), np.mean([ratio(summary, m, s, S, r) for S in RUNS])) for m in METRICS for r in RS
           for s in STRONG]
    res["L_own_budget"] = fit_L([(s, k) for s, k, _ in own], [y for *_, y in own], ema_ratio)[0] / T
    for m in METRICS:
        res[f"L_{m}"] = fit_L([(T, round(r * T)) for r in RS], [x.mean() for x in q[m]], ema_ratio)[0] / T
    if (root() / "gram").is_dir():
        res["diagnostics"] = diagnostics(*load_gram(root() / "gram")[:2])
    (root() / "fit.json").write_text(json.dumps(res, indent=1))
    print(f"L = {res['L']:.3f} T ({L:.1f} steps of 2^20 images), sse {sse:.3f}")
    print(f"own strong budget per ratio (16): L = {res['L_own_budget']:.3f} T; FID alone {res['L_fid']:.3f} T, "
          f"FD_DINOv2 alone {res['L_fd_dinov2']:.3f} T")
    for m in METRICS:
        print(f"{m:10s} measured " + " ".join(f"{v:.3f}" for v in res["targets"][m]))
    print(f"{'model':10s} fitted   " + " ".join(f"{v:.3f}" for v in res["model"]))
    if "diagnostics" in res:
        g = res["diagnostics"]
        print(f"seed noise: memory {np.round(g['memory'], 3)} -> L = {g['L_memory']:.3f} T; "
              f"spread ratio {np.round(g['spread_ratio'], 3)} -> L = {g['L_spread']:.3f} T")
    print(f"wrote {root() / 'fit.json'}")


# ------------------------------------------------------------------ CPU checks
def test(a):
    for r in RS:                                         # lambda -> 1: the plain average, sqrt((1 + r) / (1 - r))
        assert abs(ema_ratio(T, round(r * T), 1 - 1e-9) - np.sqrt((1 + r) / (1 - r))) < 1e-4
    w0 = 2.7                                             # the vertex of an exact quadratic in log w
    assert abs(optimum(WS, [3 + (np.log(w) - np.log(w0)) ** 2 for w in WS])[0] - w0) < 1e-9
    assert optimum(WS, list(WS))[0] == 1 and optimum(WS, [-w for w in WS])[0] == 8
    pts = [(T, round(r * T)) for r in RS] * 2           # a planted L comes back
    assert abs(fit_L(pts, [ema_ratio(s, k, np.exp(-1 / 50)) for s, k in pts], ema_ratio)[0] - 50) < 0.11
    rng = np.random.default_rng(0)                       # snapshots that are moving averages of random targets
    L0, d = 30.0, 4000
    lam = np.exp(-1 / L0)
    x = {r: rng.standard_normal((T, d)) for r in RUNS}
    D = {f"{model(r, k)}@{EMA}": ema_weights(k, lam) @ x[r][:k] for r in RUNS for k in BUDGETS}
    names = list(D)
    Y = np.stack([D[n] for n in names])
    g = diagnostics(names, (Y @ Y.T / d)[None])
    assert abs(g["L_memory"] * T / L0 - 1) < 0.15 and abs(g["L_spread"] * T / L0 - 1) < 0.15, g
    assert len(pairs()) == 32 and len(arms()) == 4 + 32 * (len(WS) - 1)
    print("nested_in64 test: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["gram", "sweep", "analyze", "fit", "pairs", "test"])
    ap.add_argument("--smoke", action="store_true", help="gram, sweep: NVIDIA's snapshots, a few images")
    ap.add_argument("--shard", type=int, default=0, help="gram: levels j with j %% shards == shard")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--m", type=int, default=M_GRAM, help="gram: training images")
    ap.add_argument("--batch", type=int, default=128, help="gram: images per network call")
    ap.add_argument("--task", type=int, default=0, help="sweep: arms task, task + tasks, ...")
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="sweep: keep each arm's samples and features")
    a = ap.parse_args()
    if a.stage == "pairs":
        for p in pairs():
            print(*p)
        return
    {"gram": gram, "sweep": sweep, "analyze": analyze, "fit": fit, "test": test}[a.stage](a)


if __name__ == "__main__":
    main()
