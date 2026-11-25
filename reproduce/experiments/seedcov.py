#!/usr/bin/env python
"""Cov(D_c) against Cov(D_u) along unguided SD 3.5 Medium trajectories of a single prompt.

One prompt, N initial noises (seeds 0..N-1, the repo's per-row sampling noise), sampled with the released 40-step
flow-matching Euler sampler at w = 1: the step uses the conditional prediction alone, and the latent stays in
float32 between steps as in the paper's runs. At every step k both predictions are taken on the same x_k, formed
as SD3Family.denoise_pair forms them:

    D_c = x_k - s_k v(x_k; prompt),   D_u = x_k - s_k v(x_k; ""),   x_{k+1} = x_k + (s_{k+1} - s_k) v(x_k; prompt)

Covariances are over seeds. The channel covariance is the 16 x 16 covariance of the latent's channel vector,
centred per position on the mean over seeds and pooled over the 60 x 60 positions:

    Cov(u)[k] = sum_{n,p} (u_np - ubar_p)(u_np - ubar_p)^T / ((N - 1) P)

    python reproduce/experiments/seedcov.py work    --out DIR [--prompt TEXT --n 1000 --batch 25]   # GPU
    python reproduce/experiments/seedcov.py analyze --out DIR                                       # CPU
    python reproduce/experiments/seedcov.py full    --out DIR [--k 50]   # CPU, full-dimensional PCs from grams.npz
    python reproduce/experiments/seedcov.py x0      --out DIR [--k 50]   # CPU, D_c, D_u, D_c - D_u against x_0
    python reproduce/experiments/seedcov.py align   --out DIR            # CPU, subspace overlap and covariance cosine
    python reproduce/experiments/seedcov.py alignw  --out DIR [--w 4.5]  # CPU, the same for D_w against x_0
    python reproduce/experiments/seedcov.py figures --out DIR                                       # CPU, figures only

`work` claims batches of seeds (DIR/claims/) until none are left, so any number of jobs on any partitions can share
one run. A batch whose claimant is no longer running (preempted, requeued, dead) is taken over. The job that
finds every batch done cancels its still-pending siblings (same Slurm job name), so an `afterany` dependency on
all of them starts the CPU stage.

DIR holds:
    meta.json                     prompt, N, batch, the levels (s, t, sigma)
    batches/b<j>-sums.npz         seeds, final latents, G [S, 3C, 3C] = sum over seeds and positions of u u^T and
                                  m1 [S, 3C, P] = sum over seeds of u, with u = (x_k, D_c, D_u) stacked on channels
    batches/b<j>-Dc.npy, -Du.npy  raw fields [S, B, C, H, W] float32, step-major; load_field() joins the batches
    images/<seed>.png             the decoded samples
    cov.npz                       pooled channel covariances [S, 3C, 3C] (blocks x, c, u), from all seeds and from
                                  two seed halves (even and odd batches)
    cov_pos.npy                   per-position covariances [S, P, 2C, 2C] float32 of (D_c, D_u)
    grams.npz                     full-dimensional Grams [S, N, N] of the per-coordinate-centred D_c and D_u (cc, uu,
                                  cu): Cov = X^T X / (N - 1), so these carry every full-dimensional spectrum
    analysis.npz                  eigenpairs, PC cosines, fitted scales, alignment errors, generalized eigenvalues
    pca_full-k<K>.npz             (stage full) every full-dimensional eigenvalue [S, N] per branch, and for the top K
                                  (at most N - 1): the PCs as seed weights A [S, N, K] (PC i = X^T A[:, i], X the
                                  centred raw field), their overlap [S, K, K] across branches and their stability
                                  between seed halves (NaN past the halves' own N/2 - 1 PCs)
    grams_x0.npz                  (stage x0) Grams [S, N, N] of D_c, D_u and D_c - D_u against x_0, the final latents
                                  (c0, u0, d0), of D_c - D_u with itself (dd), and x_0's own [N, N] (x00)
    pca_x0-<pair>-k<K>.npz        (stage x0) pair_pca of D_c, D_u or D_c - D_u (pair Dc, Du, Dc-Du) against x_0
    align.npz                     (stage align) per pair (DcDu, Dc, Du: D_c vs D_u, D_c vs x_0, D_u vs x_0) and seed
                                  condition (ALIGN_CONDS): <pair>_<cond>_scov [S] the covariance cosine, _a [S] the
                                  scale a*, _sub [S, K] the subspace overlap S_k for k = 1..K; <pair>_tr the trace ratio;
                                  for K in ELL_K, <pair>_{ang,pick,plane}_k<K>: the principal angles of the two top-K
                                  subspaces and both covariances in one principal plane (principal_plane)
    align_w<w>.npz                (stage alignw) the same for pair Dw, D_w = w D_c + (1 - w) D_u against x_0 (D_w
                                  evaluated on these w = 1 trajectories), principal planes included
    fig/                          the figures
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROMPT = "A red and yellow bus parked in front of a tall building."      # COCO calibration row 0
C, H = 16, 60
P = H * H
NORMS = ("fro", 2, "nuc")


# --- bookkeeping ---------------------------------------------------------------------------------------------
def nbatches(meta):
    return -(-meta["n"] // meta["batch"])


def seeds_of(meta, j):
    return list(range(j * meta["batch"], min((j + 1) * meta["batch"], meta["n"])))


def bpath(out, j, what):
    return Path(out) / "batches" / f"b{j:03d}-{what}"


def done(out, j):
    return bpath(out, j, "done").exists()


def atomic(path, write):
    """write(f) into a temporary file next to `path`, then rename it into place (no truncated files on preemption)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def me():
    return os.environ.get("SLURM_JOB_ID") or f"local-{socket.gethostname()}-{os.getpid()}"


def running(owner):
    """Whether Slurm job `owner` is running now. True when that cannot be told, so a claim moves only on evidence."""
    if not owner.isdigit():
        return True
    try:
        r = subprocess.run(["squeue", "-h", "-j", owner, "-o", "%T"], capture_output=True, text=True, timeout=60)
    except Exception:
        return True
    if r.returncode != 0:                   # squeue says "Invalid job id" once the job has left the queue
        return "Invalid job id" not in r.stderr
    return r.stdout.strip() == "RUNNING"


def claim(out, meta, stale=1800):
    """The next batch for this job: an unclaimed one, else one whose claimant is not running or whose claim is older
    than `stale` seconds. None when every batch is done or in the hands of a running job."""
    cdir = Path(out) / "claims"
    cdir.mkdir(exist_ok=True)
    held = []
    for j in range(nbatches(meta)):
        if done(out, j):
            continue
        try:
            fd = os.open(cdir / f"b{j:03d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held.append(j)
            continue
        os.write(fd, me().encode())
        os.close(fd)
        return j
    for j in held:
        c = cdir / f"b{j:03d}"
        if done(out, j) or not c.exists():
            continue
        owner, age = c.read_text().strip(), time.time() - c.stat().st_mtime
        if owner == me() or age > stale or not running(owner):
            c.write_text(me())
            print(f"taking batch {j} over from {owner} (claim {age:.0f} s old)", flush=True)
            return j
    return None


def finish(out, meta):
    left = [j for j in range(nbatches(meta)) if not done(out, j)]
    if left:
        print(f"nothing left to claim; batches {left} are in the hands of running jobs", flush=True)
        return
    # only this job array's own pending tasks: a job name can be shared by unrelated jobs (run_all.py's launcher
    # gives every job one)
    arr = os.environ.get("SLURM_ARRAY_JOB_ID")
    print("every batch is done" + (f"; cancelling the pending tasks of array {arr}" if arr else ""), flush=True)
    if arr:
        subprocess.run(["scancel", "--state", "PENDING", arr], check=False)


def load_field(out, name, step=None):
    """Raw D_c ("Dc") or D_u ("Du") of every seed: [S, N, C, H, W], or [N, C, H, W] at one step (memory-mapped)."""
    meta = json.load(open(Path(out) / "meta.json"))
    parts = [np.load(bpath(out, j, f"{name}.npy"), mmap_mode="r") for j in range(nbatches(meta))]
    if step is not None:
        return np.concatenate([a[step] for a in parts])
    return np.concatenate(parts, axis=1)


# --- GPU stage -----------------------------------------------------------------------------------------------
def trajectory(m, z, cond, each=None):
    """Final latent from initial noise z [B, C, H, W] at w = 1. each(k, x_k, D_c, D_u) sees every step (D_u is only
    computed when `each` is given)."""
    import torch
    emb, pool = (v.expand(len(z), *v.shape[1:]) for v in cond)
    ne, npl = (v.expand(len(z), *v.shape[1:]) for v in m.null)
    sig = m.pipe.scheduler.sigmas.to("cuda", torch.float32)     # the step's own float32 grid; the last entry is 0
    x = z.to("cuda", m.dtype).float()                           # the pipeline rounds the initial noise to bf16 first
    with torch.no_grad():
        for k in range(len(m.s)):
            s = torch.tensor(float(m.s[k]), device=x.device)
            v1 = m.velocity(x, k, emb, pool).float()
            if each is not None:
                v0 = m.velocity(x, k, ne, npl).float()
                each(k, x, x - s * v1, x - s * v0)
            x = x + (sig[k + 1] - sig[k]) * v1
    return x


def check(m, prompt):
    """The loop above equals the released pipeline at guidance_scale 1 with a float32 state, on seed 0."""
    import torch
    from vmg.models.sd import Fp32State
    z = m.initial_noise([0])
    with torch.no_grad(), Fp32State(m.pipe):
        ref = m.pipe(prompt=[prompt], height=m.size, width=m.size, num_inference_steps=m.spec.steps,
                     guidance_scale=1.0, max_sequence_length=m.max_tokens, latents=z.to("cuda", m.dtype),
                     output_type="latent").images.float()
    x = trajectory(m, z, m.encode([prompt]))
    rel = float((x - ref).norm() / ref.norm())
    print(f"check: this loop against the pipeline at guidance 1, seed 0: relative L2 {rel:.2e}, "
          f"max |diff| {float((x - ref).abs().max()):.2e}", flush=True)
    assert rel < 1e-3, "the sampling loop does not reproduce the released pipeline"


def run_batch(m, out, meta, j, cond):
    import torch
    from vmg import runs
    seeds = seeds_of(meta, j)
    B, S = len(seeds), len(m.s)
    t0 = time.time()
    G = torch.zeros(S, 3 * C, 3 * C, dtype=torch.float64, device="cuda")
    m1 = torch.zeros(S, 3 * C, P, dtype=torch.float64, device="cuda")
    Dc = np.empty((S, B, C, H, H), np.float32)
    Du = np.empty_like(Dc)

    def each(k, x, dc, du):
        u = torch.cat([x, dc, du], 1).reshape(B, 3 * C, P).double()
        G[k] = torch.einsum("bcp,bdp->cd", u, u)
        m1[k] = u.sum(0)
        Dc[k], Du[k] = dc.cpu().numpy(), du.cpu().numpy()

    x = trajectory(m, m.initial_noise(seeds), cond, each)
    with torch.no_grad():
        ims = np.concatenate([m.decode(x[i:i + 5]) for i in range(0, B, 5)])
    sec, gpu = time.time() - t0, torch.cuda.get_device_name()
    # the small sums first (every pooled covariance needs only them), then the raw fields, then the marker
    atomic(bpath(out, j, "sums.npz"), lambda f: np.savez(f, seeds=np.array(seeds), x_final=x.cpu().numpy(),
                                                         G=G.cpu().numpy(), m1=m1.cpu().numpy()))
    atomic(bpath(out, j, "Dc.npy"), lambda f: np.save(f, Dc))
    atomic(bpath(out, j, "Du.npy"), lambda f: np.save(f, Du))
    for sd, im in zip(seeds, ims):
        runs.save_image(im, Path(out) / "images" / f"{sd:04d}.png")
    bpath(out, j, "done").write_text(json.dumps(dict(job=me(), host=socket.gethostname(), gpu=gpu, seconds=sec)))
    print(f"batch {j}: seeds {seeds[0]}-{seeds[-1]} in {sec:.0f} s on {gpu}", flush=True)


def work(a):
    out = Path(a.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    want = dict(prompt=a.prompt, n=a.n, batch=a.batch, model="sd35m", seed=0)
    mp = out / "meta.json"
    if not mp.exists():
        atomic(mp, lambda f: f.write(json.dumps(want, indent=1).encode()))
    meta = json.load(open(mp))
    assert {k: meta[k] for k in want} == want, f"{out} holds a different run: {meta}"
    if all(done(out, j) for j in range(nbatches(meta))):
        return finish(out, meta)

    import torch
    from vmg.models.sd import SD35Medium
    t0 = time.time()
    m = SD35Medium()
    m.load()
    print(f"{socket.gethostname()} {torch.cuda.get_device_name()}: loaded in {time.time() - t0:.0f} s", flush=True)
    if "s" not in meta:
        lv = m.levels()
        meta.update(s=[float(v) for v in m.s], t=[float(v) for v in lv["t"]], sigma=[float(v) for v in lv["sigma"]])
        atomic(mp, lambda f: f.write(json.dumps(meta, indent=1).encode()))
    check(m, meta["prompt"])
    cond = m.encode([meta["prompt"]])
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):      # 11 GB the loop never uses
        setattr(m.pipe, name, None)
    torch.cuda.empty_cache()
    while (j := claim(out, meta)) is not None:
        run_batch(m, out, meta, j, cond)
    finish(out, meta)


# --- the analysis (pure numpy; tests/test_seedcov.py) --------------------------------------------------------
def cov_from_sums(G, m1, n, p=P):
    """Channel covariance [.., D, D], centred per position: G = sum over seeds and positions of u u^T, m1 [.., D, p] =
    sum over seeds of u."""
    return (G - m1 @ np.swapaxes(m1, -1, -2) / n) / ((n - 1) * p)


def eig_desc(M):
    w, V = np.linalg.eigh(M)
    return w[..., ::-1], V[..., ::-1]


def gen_eig(A, B):
    """Eigenvalues of B^-1 A (equivalently of B^-1/2 A B^-1/2), descending."""
    import scipy.linalg
    return scipy.linalg.eigvalsh(A, B)[::-1]


def fit_scale(A, B, norm):
    """(a, ||A - a B|| / ||B||) with a = argmin ||A - a B|| in the given norm ("fro", 2 = spectral, "nuc").
    For PSD A, B the minimiser lies between the extreme generalized eigenvalues, where the norm is convex in a."""
    if norm == "fro":
        a = float(np.sum(A * B) / np.sum(B * B))
    else:
        from scipy.optimize import minimize_scalar
        g = gen_eig(A, B)
        lo, hi = g.min(), g.max()
        res = minimize_scalar(lambda t: np.linalg.norm(A - t * B, norm), bounds=(lo, hi), method="bounded",
                              options=dict(xatol=1e-12 * max(hi, 1e-30)))
        a = float(res.x)
    return a, float(np.linalg.norm(A - a * B, norm) / np.linalg.norm(B, norm))


def compare(Cc, Cu):
    """Every per-step quantity of the figures for one pair of C x C covariances."""
    lc, Vc = eig_desc(Cc)
    lu, Vu = eig_desc(Cu)
    O = np.abs(Vc.T @ Vu)                              # |cos| between PC i of Cov(D_c) and PC j of Cov(D_u)
    g = gen_eig(Cc, Cu)
    out = dict(lam_c=lc, lam_u=lu, V_c=Vc, V_u=Vu, overlap=O, cos=np.diag(O).copy(), gen=g,
               white=float(g.std() / g.mean()),        # ||W - abar I||_F / ||abar I||_F, W = Cu^-1/2 Cc Cu^-1/2
               a_tr=float(np.trace(Cc) / np.trace(Cu)))
    out["err_tr"] = float(np.linalg.norm(Cc - out["a_tr"] * Cu) / np.linalg.norm(Cu))
    for nm in NORMS:
        out[f"a_{nm}"], out[f"err_{nm}"] = fit_scale(Cc, Cu, nm)
    return out


def stack(rows):
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# --- full-dimensional PCs, from the Grams alone -----------------------------------------------------------------
def gram_eig(G, K):
    """Eigenpairs of Cov = X^T X / (n - 1) from the Gram G = X X^T of n centred rows X [n, d], largest first:
    (lam [n], A [n, K]) with v_i = X^T A[:, i] the unit eigenvectors of the top K (A[:, i] = u_i / sqrt(w_i) for
    the Gram's eigenpairs (w_i, u_i)). Any inner product of such v's is then a product of Grams."""
    w, U = np.linalg.eigh(G)
    w, U = w[::-1], U[:, ::-1]
    return w / (len(G) - 1), U[:, :K] / np.sqrt(w[:K])


def recentre(G, rows, cols):
    """The Gram block of `rows` against `cols`, each side re-centred on its own subset's mean: H_r G[rows, cols] H_c."""
    g = G[np.ix_(rows, cols)]
    return g - g.mean(0, keepdims=True) - g.mean(1, keepdims=True) + g.mean()


def pair_pca(Gaa, Gbb, Gab, K, halves=None):
    """The top-K full-dimensional PCs of Cov(a) and Cov(b), two fields over the same seeds, from their Grams:
    overlap[i, j] = |v_i(Cov a) . v_j(Cov b)|, and (with two index sets `halves`) |cos| between the halves' own PC i of
    each field. n centred rows have at most n - 1 PCs, so K <= n - 1, and the halves' check stops at theirs (NaN after)."""
    assert K <= len(Gaa) - 1, f"{len(Gaa)} centred rows have only {len(Gaa) - 1} PCs, not {K}"
    la, Aa_ = gram_eig(Gaa, K)
    lb, Ab_ = gram_eig(Gbb, K)
    O = np.abs(Aa_.T @ Gab @ Ab_)
    r = dict(lam_a=la, lam_b=lb, A_a=Aa_, A_b=Ab_, overlap=O, cos=np.diag(O).copy())
    if halves is not None:
        a, b = halves
        Kh = min(K, min(len(a), len(b)) - 2)
        for name, G in (("a", Gaa), ("b", Gbb)):
            Aa, Ab = gram_eig(recentre(G, a, a), Kh)[1], gram_eig(recentre(G, b, b), Kh)[1]
            r[f"selfcos_{name}"] = np.full(K, np.nan)
            r[f"selfcos_{name}"][:Kh] = np.abs(np.diag(Aa.T @ recentre(G, a, b) @ Ab))
    return r


def full_pca(Gcc, Guu, Gcu, K, halves=None):
    """pair_pca for (D_c, D_u), with its keys named _c and _u."""
    ren = {"_a": "_c", "_b": "_u"}
    return {(k[:-2] + ren[k[-2:]] if k[-2:] in ren else k): v for k, v in pair_pca(Gcc, Guu, Gcu, K, halves).items()}


def full(a):
    """Top-K PCs of the full 16 x 60 x 60 covariances over seeds, from grams.npz; then every figure."""
    out = Path(a.out)
    meta = json.load(open(out / "meta.json"))
    N = meta["n"]
    g = np.load(out / "grams.npz")
    cc, uu, cu = g["cc"], g["uu"], g["cu"]
    half = (np.arange(N) // meta["batch"]) % 2             # the seed halves of cov.npz: even and odd batches
    halves = [np.flatnonzero(half == h) for h in (0, 1)]
    t0 = time.time()
    res = stack([full_pca(cc[k], uu[k], cu[k], a.k, halves) for k in range(len(cc))])
    for key in ("A_c", "A_u", "overlap"):                  # [S, N, K] and [S, K, K]: float32 keeps K = N - 1 near 0.5 GB
        res[key] = res[key].astype(np.float32)
    atomic(out / f"pca_full-k{a.k}.npz", lambda f: np.savez(f, K=a.k, **res))
    print(f"top-{a.k} full-dimensional PCs at {len(cc)} steps in {time.time() - t0:.0f} s", flush=True)
    figures(a)


# the pairs of stage x0: key, then label, math label and short name of each of the two fields compared
X0_PAIRS = (("Dc", "D_c", r"\Sigma_{D_c}", "c", "x_0", r"\Sigma_{x_0}", "x0"),
            ("Du", "D_u", r"\Sigma_{D_u}", "u", "x_0", r"\Sigma_{x_0}", "x0"),
            ("Dc-Du", "D_c - D_u", r"\Sigma_{D_c - D_u}", "d", "x_0", r"\Sigma_{x_0}", "x0"),
            ("Dc-Du_vs_x0-Du", "D_c - D_u", r"\Sigma_{D_c - D_u}", "d", "x_0 - D_u", r"\Sigma_{x_0 - D_u}", "r"))


def resid_grams(x00, u0, uu, d0, cu):
    """Grams of r = x_0 - D_u from those of x_0, D_c and D_u (all centred, [.., N, N]): (rr, dr) with rr = R R^T and
    dr = (X_c - X_u) R^T, R = X_0 - X_u. u0 = X_u X_0^T, d0 = (X_c - X_u) X_0^T, cu = X_c X_u^T."""
    rr = x00 - u0 - np.swapaxes(u0, -1, -2) + uu
    dr = d0 - cu + uu
    return rr, dr


def x0(a):
    """Top-K full-dimensional PCs at every step of: Cov(D_c), Cov(D_u) and Cov(D_c - D_u) against Cov(x_0), and
    Cov(D_c - D_u) against Cov(x_0 - D_u), with x_0 the final latents of the same trajectories. The Grams against x_0
    come from the raw fields once (grams_x0.npz); those of x_0 - D_u are sums of saved Grams (resid_grams)."""
    out = Path(a.out)
    meta = json.load(open(out / "meta.json"))
    N, S = meta["n"], len(meta["s"])
    gp = out / "grams_x0.npz"
    if not gp.exists():
        X0 = np.concatenate([np.load(bpath(out, j, "sums.npz"))["x_final"] for j in range(nbatches(meta))])
        X0 = X0.reshape(N, -1).astype(np.float64)
        X0 -= X0.mean(0)
        G = {k: np.zeros((S, N, N)) for k in ("c0", "u0", "dd", "d0")}
        for k in range(S):
            t0 = time.time()
            dc = load_field(out, "Dc", k).reshape(N, -1).astype(np.float64)
            du = load_field(out, "Du", k).reshape(N, -1).astype(np.float64)
            dc -= dc.mean(0)
            du -= du.mean(0)
            dd = dc - du
            G["c0"][k], G["u0"][k], G["dd"][k], G["d0"][k] = dc @ X0.T, du @ X0.T, dd @ dd.T, dd @ X0.T
            print(f"step {k + 1}: Grams against x_0 in {time.time() - t0:.0f} s", flush=True)
        atomic(gp, lambda f: np.savez(f, x00=X0 @ X0.T, **G))
    gx, g = np.load(gp), np.load(out / "grams.npz")
    x00 = gx["x00"]
    # the last Euler step lands on D_c of that step (x_0 = x - s v_c), so there D_c and x_0 must be the same field
    dev = max(np.abs(gx["c0"][-1] - x00).max(), np.abs(g["cc"][-1] - x00).max()) / np.abs(x00).max()
    print(f"last step: D_c's Grams against x_0's own, max relative difference {dev:.1e}", flush=True)
    assert dev < 1e-5, "D_c at the last step is not x_0: the fields and the final latents disagree"
    cc, uu, cu, c0, u0, dd, d0 = g["cc"], g["uu"], g["cu"], gx["c0"], gx["u0"], gx["dd"], gx["d0"]
    rr, dr = resid_grams(x00[None], u0, uu, d0, cu)
    # and there x_0 - D_u is D_c - D_u itself
    dev = max(np.abs(rr[-1] - dd[-1]).max(), np.abs(dr[-1] - dd[-1]).max()) / np.abs(dd[-1]).max()
    print(f"last step: x_0 - D_u's Grams against D_c - D_u's own, max relative difference {dev:.1e}", flush=True)
    assert dev < 1e-6, "x_0 - D_u at the last step is not D_c - D_u"
    half = (np.arange(N) // meta["batch"]) % 2
    halves = [np.flatnonzero(half == h) for h in (0, 1)]
    grams = {"Dc": (cc, x00, c0), "Du": (uu, x00, u0), "Dc-Du": (dd, x00, d0), "Dc-Du_vs_x0-Du": (dd, rr, dr)}
    for key, la, *_, lb, _, _ in X0_PAIRS:
        Gaa, Gbb, Gab = grams[key]
        t0 = time.time()
        res = stack([pair_pca(Gaa[k], Gbb if Gbb.ndim == 2 else Gbb[k], Gab[k], a.k, halves) for k in range(S)])
        for f_ in ("A_a", "A_b", "overlap"):
            res[f_] = res[f_].astype(np.float32)
        atomic(out / f"pca_x0-{key}-k{a.k}.npz", lambda f: np.savez(f, K=a.k, **res))
        print(f"{la} against {lb}: top-{a.k} PCs at {S} steps in {time.time() - t0:.0f} s", flush=True)
    figures(a)


# --- alignment without matching PCs one by one: subspace overlap and the covariance cosine -------------------------
def cov_cos(Gaa, Gbb, Gab):
    """(S_cov, a*) for Cov(a) and Cov(b) from Grams of centred rows: S_cov = tr(Sa Sb) / (|Sa|_F |Sb|_F), the cosine
    between the covariances as vectors, and a* = argmin_a |Sa - a Sb|_F = tr(Sa Sb) / |Sb|_F^2. X_a^T X_a X_b^T X_b
    has trace |X_a X_b^T|_F^2, so tr(Sa Sb) = |Gab|_F^2 / ((n_a - 1)(n_b - 1)): full-dimensional, no PC dropped."""
    na, nb, t = len(Gaa), len(Gbb), np.sum(Gab ** 2)
    return t / (np.linalg.norm(Gaa) * np.linalg.norm(Gbb)), t / np.sum(Gbb ** 2) * (nb - 1) / (na - 1)


def subspace_overlap(Aa, Ab, Gab):
    """S_k = |U_a^T U_b|_F^2 / k for every k up to the PCs given: the mean cos^2 of the principal angles between the
    two top-k subspaces, so it ignores the order of PCs within them. Aa, Ab: PCs as seed weights (gram_eig)."""
    O2 = (Aa.T @ Gab @ Ab) ** 2
    K = min(O2.shape)
    return np.diag(O2[:K, :K].cumsum(0).cumsum(1)) / np.arange(1, K + 1)


# key, then the two fields: c = D_c, u = D_u, x = x_0
ALIGN_PAIRS = (("DcDu", "c", "u"), ("Dc", "c", "x"), ("Du", "u", "x"))
# all seeds; both fields on the same half (mean of the two halves); the two fields on different halves (mean of
# both ways round); each field's two halves against each other, the estimation-noise baseline
ALIGN_CONDS = {"paired": [("all", "all")], "paired_half": [(0, 0), (1, 1)], "cross": [(0, 1), (1, 0)],
               "self_a": [(0, 1)], "self_b": [(0, 1)]}
ELL_K = (50, 100)                                           # stage alignw: the subspaces drawn as ellipses


def guided_grams(w, cc, uu, cu, c0, u0):
    """Grams of D_w = w D_c + (1 - w) D_u, centred as D_c and D_u are: with itself, and against x_0 (rows D_w)."""
    return w * w * cc + (1 - w) ** 2 * uu + w * (1 - w) * (cu + cu.T), w * c0 + (1 - w) * u0


def principal_plane(Aa, Ab, Gaa, Gbb, Gab, i=None, out_a=None):
    """The principal angles between the top-K subspaces of Cov(a) and Cov(b) (PCs as seed weights, K columns each),
    and both covariances projected onto the plane of principal-vector pair i: p_i in a's subspace, q_i in b's,
    p_i . q_i = cos theta_i. The plane's axes are p_i and the part of q_i orthogonal to it. That second axis is
    orthogonal to all of a's subspace, and the in-plane normal of q_i to all of b's, so a's ellipse has its long axis
    exactly on the first axis and b's exactly at theta_i. Default i: the pair whose cos^2 is nearest S_K, their mean.
    When q_i = p_i (theta_i = 0: the same field, e.g. D_c against x_0 at the last step) that plane does not exist, and
    the second axis is out_a instead (seed weights of a unit vector outside a's top K, e.g. its PC K + 1, so the plane
    is built as the others are), or without it the next principal vector of a.
    Returns (cos [K] descending, i, Sa [2, 2], Sb [2, 2])."""
    Y, cs, Zt = np.linalg.svd(Aa.T @ Gab @ Ab)
    cs = np.clip(cs, 0, 1)
    if i is None:
        i = int(np.argmin(np.abs(cs ** 2 - np.mean(cs ** 2))))
    G = {"aa": Gaa, "ab": Gab, "ba": Gab.T, "bb": Gbb}
    n = {"a": len(Gaa), "b": len(Gbb)}

    def proj(vecs, f):                                      # [u_j . Cov(f) u_k] for u = X_side^T weights
        return np.array([[g1 @ G[s1 + f] @ G[f + s2] @ g2 for s2, g2 in vecs] for s1, g1 in vecs]) / (n[f] - 1)

    c = cs[i]
    s = np.sqrt(1 - c * c)
    if s < 1e-6:
        e2 = out_a if out_a is not None else Aa @ Y[:, i + 1 if i + 1 < len(cs) else i - 1]
        vecs = [("a", Aa @ Y[:, i]), ("a", e2)]             # already orthonormal
        return cs, i, proj(vecs, "a"), proj(vecs, "b")
    vecs = [("a", Aa @ Y[:, i]), ("b", Ab @ Zt[i])]         # p_i = X_a^T (Aa Y_i), q_i = X_b^T (Ab Z_i)
    T = np.array([[1, -c / s], [0, 1 / s]])                 # [e1 e2] = [p q] T
    return cs, i, T.T @ proj(vecs, "a") @ T, T.T @ proj(vecs, "b") @ T


def align_steps(out, pairs, w=None, planes=()):
    """S_k (every k), (S_cov, a*) and the trace ratio at every step for `pairs` (key, field a, field b), over all
    seeds and over the two seed halves (ALIGN_CONDS), from grams.npz and grams_x0.npz. Fields: c = D_c, u = D_u,
    x = x_0, and with w given, w = D_w. With `planes`, also principal_plane on all seeds for each pair and each K in
    planes, with the pair's second field as the reference (principal_plane's a): <pair>_ang_k<K> [S, K] the angles in
    degrees, <pair>_pick_k<K> [S] the pair drawn, <pair>_plane_k<K> [S, 2, 2, 2] the two projected covariances (the
    second field's, then the first's)."""
    meta = json.load(open(out / "meta.json"))
    N, S = meta["n"], len(meta["s"])
    half = (np.arange(N) // meta["batch"]) % 2             # the seed halves of cov.npz: even and odd batches
    rows = {"all": np.arange(N), 0: np.flatnonzero(half == 0), 1: np.flatnonzero(half == 1)}
    g, gx = np.load(out / "grams.npz"), np.load(out / "grams_x0.npz")
    cc, uu, cu, c0, u0, x00 = g["cc"], g["uu"], g["cu"], gx["c0"], gx["u0"], gx["x00"]
    res = {f"{p}_{c}_{q}": [] for p, _, _ in pairs for c in ALIGN_CONDS for q in ("scov", "a", "sub")}
    res.update({f"{p}_tr": [] for p, _, _ in pairs})
    res.update({f"{p}_{q}_k{K}": [] for p, _, _ in pairs for K in planes for q in ("ang", "pick", "plane")})
    pcs = {}                                                # (field, rows) -> PCs as seed weights
    t0 = time.time()
    for k in range(S):
        G = {"cc": cc[k], "uu": uu[k], "cu": cu[k], "xx": x00, "cx": c0[k], "ux": u0[k]}
        if w is not None:
            G["ww"], G["wx"] = guided_grams(w, cc[k], uu[k], cu[k], c0[k], u0[k])
        gram = lambda f1, f2: G[f1 + f2] if f1 + f2 in G else G[f2 + f1].T           # noqa: E731
        block = lambda f1, r1, f2, r2: (gram(f1, f2) if r1 == r2 == "all"             # noqa: E731
                                        else recentre(gram(f1, f2), rows[r1], rows[r2]))
        for key in [key for key in pcs if key[0] != "x"]:  # x_0's PCs do not change with the step: kept
            del pcs[key]

        def pc(f, r):
            if (f, r) not in pcs:
                Gr = block(f, r, f, r)
                pcs[f, r] = gram_eig(Gr, len(Gr) - (1 if r == "all" else 2))[1]
            return pcs[f, r]

        for p, fa, fb in pairs:
            for cond, ways in ALIGN_CONDS.items():
                f1, f2 = {"self_a": (fa, fa), "self_b": (fb, fb)}.get(cond, (fa, fb))
                got = []
                for r1, r2 in ways:
                    Gab = block(f1, r1, f2, r2)
                    got.append((*cov_cos(block(f1, r1, f1, r1), block(f2, r2, f2, r2), Gab),
                                subspace_overlap(pc(f1, r1), pc(f2, r2), Gab)))
                res[f"{p}_{cond}_scov"].append(np.mean([q[0] for q in got]))
                res[f"{p}_{cond}_a"].append(np.mean([q[1] for q in got]))
                res[f"{p}_{cond}_sub"].append(np.mean([q[2] for q in got], 0))
            res[f"{p}_tr"].append(np.trace(gram(fa, fa)) / np.trace(gram(fb, fb)))
        for (p, fa, fb), K in [(pair, K) for pair in pairs for K in planes]:
            cs, i, Sref, Sf = principal_plane(pc(fb, "all")[:, :K], pc(fa, "all")[:, :K], gram(fb, fb), gram(fa, fa),
                                              gram(fb, fa), out_a=pc(fb, "all")[:, K])
            res[f"{p}_ang_k{K}"].append(np.degrees(np.arccos(cs)))
            res[f"{p}_pick_k{K}"].append(i)
            res[f"{p}_plane_k{K}"].append(np.stack([Sref, Sf]))
        p = pairs[0][0]
        print(f"step {k + 1}: {p} S_cov {res[f'{p}_paired_scov'][-1]:.4f}, a* {res[f'{p}_paired_a'][-1]:.4f}, "
              f"S_50 {res[f'{p}_paired_sub'][-1][:50][-1]:.3f} ({time.time() - t0:.0f} s)", flush=True)
    return {key: np.array(v) for key, v in res.items()}


def align(a):
    """align_steps for D_c vs D_u, D_c vs x_0 and D_u vs x_0 (ALIGN_PAIRS), with the principal planes for K in ELL_K
    -> align.npz; then every figure."""
    out = Path(a.out)
    res = align_steps(out, ALIGN_PAIRS, planes=ELL_K)
    # the last Euler step lands on D_c (see x0()), so there D_c against x_0 is a field against itself
    assert abs(res["Dc_paired_scov"][-1] - 1) < 1e-6 and abs(res["Dc_paired_sub"][-1][:500] - 1).max() < 1e-4
    atomic(out / "align.npz", lambda f: np.savez(f, **res))
    figures(a)


def alignw(a):
    """align_steps for D_w = w D_c + (1 - w) D_u against x_0, on the same unguided trajectories (D_w is only
    evaluated there, never stepped with), with the principal planes for K = 50 and 100 -> align_w<w>.npz."""
    out = Path(a.out)
    res = align_steps(out, (("Dw", "w", "x"),), w=a.w, planes=ELL_K)
    atomic(out / f"align_w{a.w:g}.npz", lambda f: np.savez(f, w=a.w, **res))
    figures(a)


def analyze(a):
    out = Path(a.out)
    meta = json.load(open(out / "meta.json"))
    nb, N = nbatches(meta), meta["n"]
    missing = [j for j in range(nb) if not done(out, j)]
    assert not missing, f"batches not done: {missing}"
    # pooled channel covariances from the per-batch sums, for all seeds and for the two halves
    acc = {h: [0.0, 0.0, 0] for h in (0, 1)}
    for j in range(nb):
        z = np.load(bpath(out, j, "sums.npz"))
        h = acc[j % 2]
        h[0], h[1], h[2] = h[0] + z["G"], h[1] + z["m1"], h[2] + len(z["seeds"])
    cov = cov_from_sums(acc[0][0] + acc[1][0], acc[0][1] + acc[1][1], acc[0][2] + acc[1][2])
    halves = np.stack([cov_from_sums(*acc[h]) for h in (0, 1)])
    assert acc[0][2] + acc[1][2] == N
    S = cov.shape[0]
    atomic(out / "cov.npz", lambda f: np.savez(f, cov=cov, halves=halves, blocks=np.array(["x", "c", "u"]), n=N,
                                               s=np.array(meta["s"]), t=np.array(meta["t"])))
    blk = lambda M, i, j: M[..., i * C:(i + 1) * C, j * C:(j + 1) * C]       # noqa: E731  0 = x, 1 = D_c, 2 = D_u
    full = stack([compare(blk(cov[k], 1, 1), blk(cov[k], 2, 2)) for k in range(S)])
    half = [stack([compare(blk(halves[h, k], 1, 1), blk(halves[h, k], 2, 2)) for k in range(S)]) for h in (0, 1)]
    # how well each PC is pinned down by the seeds: |cos| between the two halves' PC i, per branch
    selfcos = {b: np.abs(np.einsum("sci,sci->si", half[0][f"V_{b}"], half[1][f"V_{b}"])) for b in "cu"}
    lam_x = eig_desc(blk(cov, 0, 0))[0]
    res = dict(full, lam_x=lam_x, selfcos_c=selfcos["c"], selfcos_u=selfcos["u"],
               **{f"h{h}_{k}": half[h][k] for h in (0, 1) for k in ("cos", "gen", "white", "a_tr", "err_tr",
                                                                    *(f"a_{n}" for n in NORMS),
                                                                    *(f"err_{n}" for n in NORMS))})
    atomic(out / "analysis.npz", lambda f: np.savez(f, **res))
    print("pooled covariances and their comparison written", flush=True)

    # per-position covariances and full-dimensional Grams from the raw fields
    tmp = out / ".cov_pos.tmp.npy"
    cp = np.lib.format.open_memmap(tmp, "w+", np.float32, (S, P, 2 * C, 2 * C))
    grams = {k: np.zeros((S, N, N)) for k in ("cc", "uu", "cu")}
    for k in range(S):
        t0 = time.time()
        dc = load_field(out, "Dc", k).reshape(N, C, P).astype(np.float64)
        du = load_field(out, "Du", k).reshape(N, C, P).astype(np.float64)
        dc -= dc.mean(0)
        du -= du.mean(0)
        u = np.ascontiguousarray(np.concatenate([dc, du], 1).transpose(2, 0, 1))      # [P, N, 2C]
        cp[k] = (np.swapaxes(u, 1, 2) @ u / (N - 1)).astype(np.float32)
        pooled = cp[k].astype(np.float64).mean(0)
        ref = np.block([[blk(cov[k], 1, 1), blk(cov[k], 1, 2)], [blk(cov[k], 2, 1), blk(cov[k], 2, 2)]])
        dev = np.abs(pooled - ref).max() / np.abs(ref).max()
        assert dev < 1e-4, f"step {k}: raw fields and sums disagree ({dev:.1e})"
        Xc, Xu = dc.reshape(N, -1), du.reshape(N, -1)
        grams["cc"][k], grams["uu"][k], grams["cu"][k] = Xc @ Xc.T, Xu @ Xu.T, Xc @ Xu.T
        print(f"step {k}: per-position covariances and Grams in {time.time() - t0:.0f} s "
              f"(raw vs sums {dev:.1e})", flush=True)
    cp.flush()
    del cp
    os.replace(tmp, out / "cov_pos.npy")
    atomic(out / "grams.npz", lambda f: np.savez(f, **grams))
    figures(a)


# --- figures -------------------------------------------------------------------------------------------------
def vgap_seeds(G, m1, n, s, c=C):
    """fig:vgap's quantity on this run: at each step the mean over seeds of |v_c - v_u|^2 / d, with
    v_c - v_u = (D_u - D_c) / s, from the per-batch sums (G = sum of u u^T over seeds and positions, m1 = sum of u
    over seeds, u = (x, D_c, D_u) on channels). Returns (total, mean part |E(v_c - v_u)|^2 / d, the rest: the spread
    over seeds), each [S]; total = mean part + spread."""
    cb, ub = slice(c, 2 * c), slice(2 * c, 3 * c)
    d = c * m1.shape[-1]
    tot = np.trace(G[:, cb, cb] + G[:, ub, ub] - G[:, cb, ub] - G[:, ub, cb], axis1=1, axis2=2) / (n * d)
    mean = np.sum(((m1[:, cb] - m1[:, ub]) / n) ** 2, axis=(1, 2)) / d
    return tot / s ** 2, mean / s ** 2, (tot - mean) / s ** 2


def vgap_run(out):
    """vgap_seeds for the run in out, cached as out/vgap.npz (from batches/b*-sums.npz the first time); None
    without either."""
    f = out / "vgap.npz"
    if not f.exists():
        parts = sorted((out / "batches").glob("b*-sums.npz"))
        if not parts:
            return None
        G = m1 = 0
        n = 0
        for q in parts:
            z = np.load(q)
            G, m1, n = G + z["G"], m1 + z["m1"], n + len(z["seeds"])
        s = np.array(json.load(open(out / "meta.json"))["s"])
        tot, mean, spread = vgap_seeds(G, m1, n, s)
        atomic(f, lambda fh: np.savez(fh, total=tot, mean=mean, spread=spread, n=n, s=s))
    return np.load(f)


def vgap_curve():
    """fig:vgap's curve, E||v_c - v_u||^2 / d at each step of the 40-step sampler (SD 3.5 Medium on the 5,705 COCO
    calibration photos), and E||v*||^2 / d for its right axis: from the moments pass when it is here, as fig_vgap.py
    reads it, else from its reduced copy tests/fixtures/sd35m-coco-s40.npz. That copy has the same rr but only x_0's
    centred variance, not E||x_0||^2, so it gives no E||v*||^2 (None)."""
    from vmg import runs
    if runs.moments_shards("sd35m", "coco"):
        from reproduce.figures.fig_vgap import from_moments
        _, gap, ref = from_moments()
        return gap, ref
    z = np.load(REPO / "tests" / "fixtures" / "sd35m-coco-s40.npz")
    return (z["rr"][:, 2] - 2 * z["rr"][:, 1] + z["rr"][:, 0]) / z["t"] ** 2, None


NORM_STYLE = {"fro": ("Frobenius", "#6250d6", "-"), 2: ("spectral", "#eda100", "--"),
              "nuc": ("nuclear", "#e34948", ":"), "tr": ("trace ratio", "#008300", "-.")}
SEQ = ["#f5f8fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # one hue, light = 0
PC_RAMP = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"]                           # PC1 dark -> PC16 light


def figures(a):
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Ellipse
    from reproduce import style as st

    plt = st.rc(size=9, small=8)
    out = Path(a.out)
    fd = out / "fig"
    fd.mkdir(exist_ok=True)
    meta = json.load(open(out / "meta.json"))
    A = np.load(out / "analysis.npz")
    cov = np.load(out / "cov.npz")["cov"]
    S = cov.shape[0]
    ks, s = np.arange(1, S + 1), np.array(meta["s"])        # sampling step 1..S, as in fig:vgap; level k = step - 1
    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    ramp = LinearSegmentedColormap.from_list("pc", PC_RAMP)
    pcol = [ramp(i / (C - 1)) for i in range(C)]
    sub = f'SD 3.5 Medium, w = 1, {meta["n"]} seeds, prompt "{meta["prompt"]}"'

    def head(ax, text):
        ax.text(0, 1.2 if getattr(ax, "_has_top", False) else 1.04, text, transform=ax.transAxes, fontsize=8.5,
                color=st.INK, va="bottom")

    def step_axis(ax, top=True):
        ax._has_top = top
        ax.set_xlim(0.5, S + 0.5)
        ax.set_xlabel("sampling step  (1 = pure noise)")
        if top:
            sec = ax.secondary_xaxis("top", functions=(lambda k: np.interp(k, ks, s),
                                                       lambda v: np.interp(v, s[::-1], ks[::-1])))
            sec.set_xticks([1.0, 0.9, 0.75, 0.5, 0.25, 0.1])
            sec.set_xlabel("flow time s", color=st.MUTED)
            sec.tick_params(colors=st.MUTED, labelcolor=st.MUTED, width=0.6, length=2.5)
            sec.spines["top"].set_color(st.MUTED)

    # fig:vgap's zones: the guidance interval of Kynkaanniemi et al. (2024) in this sampler's steps
    from reproduce.figures.fig_vgap import SIG_HI, SIG_LO
    with np.errstate(divide="ignore"):
        tlev = s / (1 - s)                                   # the paper's noise level t (EDM sigma)
    on = (tlev > SIG_LO) & (tlev <= SIG_HI)
    first, last = ks[on][0], ks[on][-1]
    assert on[first - 1:last].all(), "the guidance interval is not one run of steps"

    def zones(ax, shade=True, legend=False):
        """Shade fig:vgap's zones: guidance harmful (before the interval), the interval, largely unneeded (after)."""
        h = []
        if shade:
            h.append(ax.axvspan(0.5, first - 0.5, color=st.ACC, alpha=0.10, lw=0, zorder=0,
                                label=f"guidance harmful, high noise (steps 1-{first - 1})"))
            h.append(ax.axvspan(first - 0.5, last + 0.5, color="none", lw=0, zorder=0,
                                label=f"guidance interval $t\\in({SIG_LO:g}, {SIG_HI:g}]$ (steps {first}-{last}),\n"
                                      "Kynkäänniemi et al. (2024), as in fig:vgap"))
            h.append(ax.axvspan(last + 0.5, S + 0.5, color=st.MUTED, alpha=0.13, lw=0, zorder=0,
                                label=f"largely unneeded (steps {last + 1}-{S})"))
        for xb in (first - 0.5, last + 0.5):
            ax.axvline(xb, color=st.INK if shade else "#ffffff", lw=0.9 if shade else 1.3, ls=(0, (3, 2)), zorder=5)
        if legend:
            from matplotlib.patches import Patch
            hs = [Patch(facecolor=st.ACC, alpha=0.25, label=h[0].get_label()),
                  Patch(facecolor="#ffffff", edgecolor=st.MUTED, lw=0.6, label=h[1].get_label()),
                  Patch(facecolor=st.MUTED, alpha=0.3, label=h[2].get_label())]
            ax.legend(handles=hs, loc="best", frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.5)

    def pc_colorbar(fig, ax, label="PC rank i"):
        sm = cm.ScalarMappable(norm=Normalize(1, C), cmap=ramp)
        cb = fig.colorbar(sm, ax=ax, ticks=[1, 4, 8, 12, 16], fraction=0.04, pad=0.02)
        cb.set_label(label)
        cb.outline.set_visible(False)

    def save(fig, name):
        fig.savefig(fd / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(fd / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 1. PC alignment: |cos| between PC i of Cov(D_c) and PC i of Cov(D_u), as 16 curves and as a heatmap
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1, 1], wspace=0.42))
    for i in range(C):
        ax[0].plot(ks, A["cos"][:, i], color=pcol[i], lw=1.4, zorder=3 + C - i)
    ax[0].set_ylim(0, 1.03)
    ax[0].set_ylabel(r"$|\cos(v_i(\Sigma_c),\, v_i(\Sigma_u))|$")
    st.style(ax[0])
    step_axis(ax[0])
    zones(ax[0], legend=True)
    pc_colorbar(fig, ax[0])
    head(ax[0], "(a) PC i of Cov(D_c) against PC i of Cov(D_u), over fig:vgap's guidance zones")
    for j, (M, t) in enumerate([(A["cos"], "(b) the same as a heatmap (white dashes: the guidance interval)"),
                                (np.minimum(A["selfcos_c"], A["selfcos_u"]),
                                 "(c) how well PC i is pinned down: |cos| between\n     two seed halves, the worse branch")]):
        im = ax[j + 1].imshow(M.T, aspect="auto", cmap=seq, vmin=0, vmax=1, interpolation="nearest",
                              extent=(0.5, S + 0.5, C + 0.5, 0.5))
        ax[j + 1].set_ylabel("PC rank i")
        ax[j + 1].set_yticks([1, 4, 8, 12, 16])
        st.style(ax[j + 1], grid=False)
        step_axis(ax[j + 1], top=False)
        zones(ax[j + 1], shade=False)
        head(ax[j + 1], t)
    cb = fig.colorbar(im, ax=ax[1:], fraction=0.03, pad=0.02)
    cb.set_label("|cos|")
    cb.outline.set_visible(False)
    fig.suptitle(sub, y=1.14, fontsize=9, color=st.MUTED)
    save(fig, "pca_cos")

    # 2. the full PC overlap |V_c^T V_u| at a few steps
    pick = [0, 4, 10, 18, 26, 34, S - 1]
    fig, ax = plt.subplots(1, len(pick), figsize=(2.3 * len(pick), 2.7))
    for a_, k in zip(ax, pick):
        im = a_.imshow(A["overlap"][k], cmap=seq, vmin=0, vmax=1, interpolation="nearest",
                       extent=(0.5, C + 0.5, C + 0.5, 0.5))
        a_.set_xticks([1, 8, 16])
        a_.set_yticks([1, 8, 16])
        a_.set_xlabel("PC j of Cov(D_u)")
        if a_ is ax[0]:
            a_.set_ylabel("PC i of Cov(D_c)")
        st.style(a_, grid=False)
        head(a_, f"step {k + 1}, s = {s[k]:.2f}")
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label(r"$|v_i(\Sigma_c)^\top v_j(\Sigma_u)|$")
    cb.outline.set_visible(False)
    fig.suptitle(sub, y=1.1, fontsize=9, color=st.MUTED)
    save(fig, "pc_overlap")

    # 3. spectra
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a_, key, t in [(ax[0], "lam_c", "(a) eigenvalues of Cov(D_c)"), (ax[1], "lam_u", "(b) eigenvalues of Cov(D_u)")]:
        for i in range(C):
            a_.plot(ks, A[key][:, i], color=pcol[i], lw=1.4)
        a_.set_yscale("log")
        st.style(a_)
        step_axis(a_)
        head(a_, t)
    ax[0].set_ylabel("variance per channel direction")
    pc_colorbar(fig, ax[1])
    fig.suptitle(sub, y=1.16, fontsize=9, color=st.MUTED)
    save(fig, "spectra")

    # 4. alignment: Cov(D_c) = a Cov(D_u)?
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))
    a0, a1, a2, a3 = ax.ravel()
    for a_ in ax.ravel():
        st.style(a_)
        step_axis(a_, top=a_ in (a0, a1))
    for nm in (*NORMS, "tr"):
        lab, col, ls = NORM_STYLE[nm]
        a0.plot(ks, A[f"a_{nm}"], color=col, ls=ls, lw=1.6, label=lab + (" (least squares)" if nm == "fro" else ""))
        a1.plot(ks, A[f"err_{nm}"], color=col, ls=ls, lw=1.6,
                label=lab + (", Frobenius error at a = tr ratio" if nm == "tr" else ""))
        lo = np.minimum(A[f"h0_err_{nm}"], A[f"h1_err_{nm}"])
        hi = np.maximum(A[f"h0_err_{nm}"], A[f"h1_err_{nm}"])
        a1.fill_between(ks, lo, hi, color=col, alpha=0.18, lw=0)
    a0.set_ylabel(r"fitted scale $a$")
    a0.legend(frameon=False)
    head(a0, r"(a) the scale $a = \arg\min_a \|\Sigma_c - a\Sigma_u\|$ per norm, and $\mathrm{tr}\,\Sigma_c / \mathrm{tr}\,\Sigma_u$")
    a1.set_yscale("log")
    a1.set_ylabel(r"$\|\Sigma_c - a\Sigma_u\| \,/\, \|\Sigma_u\|$")
    a1.legend(frameon=False, fontsize=7.5)
    head(a1, "(b) relative alignment error, each at its own a (bands: the two seed halves)")
    for i in range(C):
        a2.plot(ks, A["gen"][:, i], color=pcol[i], lw=1.3)
    a2.plot(ks, A["a_fro"], color=st.ACC, ls="--", lw=1.6, label="a (Frobenius)")
    a2.set_yscale("log")
    a2.set_ylabel(r"eigenvalues of $\Sigma_u^{-1}\Sigma_c$")
    a2.legend(frameon=False, loc="upper left")
    pc_colorbar(fig, a2, label="rank")
    head(a2, r"(c) generalized eigenvalues: all 16 equal $a$ iff $\Sigma_c = a\Sigma_u$")
    a3.plot(ks, A["white"], color=st.INK, lw=1.6)
    a3.fill_between(ks, np.minimum(A["h0_white"], A["h1_white"]), np.maximum(A["h0_white"], A["h1_white"]),
                    color=st.INK, alpha=0.15, lw=0)
    a3.set_ylim(bottom=0)
    a3.set_ylabel(r"$\|W - \bar a I\|_F / \|\bar a I\|_F$")
    head(a3, r"(d) whitened error, $W = \Sigma_u^{-1/2}\Sigma_c\Sigma_u^{-1/2}$, $\bar a = \mathrm{tr}\,W/16$"
                 " (std / mean of (c))")
    fig.suptitle(sub, y=1.01, fontsize=9, color=st.MUTED)
    fig.tight_layout(h_pad=3)
    save(fig, "alignment")

    # 5. ellipses in the PC planes of Cov(D_u)
    pick = [0, 3, 8, 14, 20, 26, 32, S - 1]
    planes = [(0, 1), (2, 3)]
    fig, ax = plt.subplots(len(planes), len(pick), figsize=(2.25 * len(pick), 2.5 * len(planes)))
    for col, k in enumerate(pick):
        Cc, Cu = cov[k, C:2 * C, C:2 * C], cov[k, 2 * C:, 2 * C:]
        Vu = A["V_u"][k]
        for row, (i, j) in enumerate(planes):
            a_ = ax[row, col]
            Bm = Vu[:, [i, j]]
            ext = 0
            for M, ec, ls, lab in [(Cu, st.UNCOND, "-", r"$\Sigma_u$"), (Cc, st.COND, "-", r"$\Sigma_c$"),
                                   (A["a_fro"][k] * Cu, st.ACC, "--", r"$a\,\Sigma_u$")]:
                w, V = np.linalg.eigh(Bm.T @ M @ Bm)
                ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
                r = 2 * np.sqrt(np.maximum(w, 0))
                a_.add_patch(Ellipse((0, 0), 2 * r[1], 2 * r[0], angle=ang, fill=False, ec=ec, ls=ls, lw=1.5,
                                     label=lab))
                ext = max(ext, r.max())
            a_.set_xlim(-1.15 * ext, 1.15 * ext)
            a_.set_ylim(-1.15 * ext, 1.15 * ext)
            a_.set_aspect("equal")
            a_.axhline(0, color=st.GRID, lw=0.6, zorder=0)
            a_.axvline(0, color=st.GRID, lw=0.6, zorder=0)
            a_.set_xlabel(f"PC{i + 1} of Cov(D_u)", fontsize=7.5)
            a_.set_ylabel(f"PC{j + 1}", fontsize=7.5)
            st.style(a_, grid=False)
            if row == 0:
                head(a_, f"step {k + 1}, s = {s[k]:.2f}")
    ax[0, 0].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.12), ncol=3)
    fig.suptitle(sub + "; 2-sigma ellipses in Cov(D_u)'s own PC planes, a = Frobenius fit", y=1.03, fontsize=9,
                 color=st.MUTED)
    fig.tight_layout()
    save(fig, "ellipses")

    # 6. ellipsoids in the top-3 PC space of Cov(D_u)
    pick = [0, 8, 20, S - 1]
    fig = plt.figure(figsize=(4.2 * len(pick), 4.4))
    uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    sphere = np.stack([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)])         # [3, 20, 40]
    for n_, k in enumerate(pick):
        a_ = fig.add_subplot(1, len(pick), n_ + 1, projection="3d")
        Bm = A["V_u"][k][:, :3]
        ext = 0
        for M, ec, lab in [(cov[k, 2 * C:, 2 * C:], st.UNCOND, r"$\Sigma_u$"), (cov[k, C:2 * C, C:2 * C], st.COND, r"$\Sigma_c$")]:
            w, V = np.linalg.eigh(Bm.T @ M @ Bm)
            L = V @ np.diag(2 * np.sqrt(np.maximum(w, 0)))
            X = np.einsum("ij,jab->iab", L, sphere)
            a_.plot_wireframe(*X, color=ec, lw=0.5, alpha=0.7, rstride=2, cstride=2, label=lab)
            ext = max(ext, np.abs(X).max())
        for f in (a_.set_xlim, a_.set_ylim, a_.set_zlim):
            f(-ext, ext)
        a_.set_box_aspect((1, 1, 1))
        a_.set_xlabel("PC1")
        a_.set_ylabel("PC2")
        a_.set_zlabel("PC3")
        a_.set_title(f"step {k + 1}, s = {s[k]:.2f}", fontsize=9, color=st.INK)
        a_.tick_params(labelsize=6.5, colors=st.MUTED)
        if n_ == 0:
            a_.legend(frameon=False, loc="upper left")
    fig.suptitle(sub + "; 2-sigma ellipsoids in the top-3 PCs of Cov(D_u)", y=0.98, fontsize=9, color=st.MUTED)
    save(fig, "ellipsoids3d")

    # 7. the first 64 samples
    from PIL import Image
    files = [f for f in sorted((out / "images").glob("*.png")) if not f.name.startswith(".")][:64]
    if files:
        tile = 160
        sheet = Image.new("RGB", (8 * tile, 8 * tile), "white")
        for i, f in enumerate(files):
            sheet.paste(Image.open(f).convert("RGB").resize((tile, tile), Image.LANCZOS), ((i % 8) * tile, (i // 8) * tile))
        sheet.save(fd / "samples.jpg", quality=92)

    # 8. the numbers behind the figures, at a few steps
    lines = [sub, "", f"{'step':>4} {'s':>6} {'tr ratio':>9} {'a_F':>7} {'err_F':>7} {'err_2':>7} {'err_nuc':>8} "
             f"{'white':>7} {'cos1':>6} {'cos2':>6} {'cos3':>6} {'cos16':>6}"]
    for k in range(S):
        lines.append(f"{k + 1:4d} {s[k]:6.3f} {A['a_tr'][k]:9.3f} {A['a_fro'][k]:7.3f} {A['err_fro'][k]:7.3f} "
                     f"{A['err_2'][k]:7.3f} {A['err_nuc'][k]:8.3f} {A['white'][k]:7.3f} {A['cos'][k, 0]:6.3f} "
                     f"{A['cos'][k, 1]:6.3f} {A['cos'][k, 2]:6.3f} {A['cos'][k, 15]:6.3f}")
    # 9. how many of the 16 PCs match, per step: PC i of Cov(D_c) against PC i of Cov(D_u) ("same rank"), and against
    # its best PC j of Cov(D_u) ("any rank"). Above 1/sqrt(2) a best match is one-to-one, so both are counts of pairs.
    best = A["overlap"].max(axis=2)
    zone = np.where(ks < first, "harmful", np.where(ks <= last, "interval", "unneeded"))
    rows = [(int(ks[k]), float(s[k]), float(tlev[k]), str(zone[k]),
             *(int((M[k] > th).sum()) for M in (A["cos"], best) for th in (0.9, 0.8))) for k in range(S)]
    hdr = "step,s,t,zone,same_rank_gt0.9,same_rank_gt0.8,any_rank_gt0.9,any_rank_gt0.8"
    (fd / "pc_match.csv").write_text(hdr + "\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")
    lines += ["", f"PCs of Cov(D_c) matched to Cov(D_u) out of {C}, |cos| above 0.9 / 0.8 (zones as in fig:vgap)",
              f"{'step':>4} {'t':>8} {'zone':>9} {'same>.9':>8} {'same>.8':>8} {'any>.9':>7} {'any>.8':>7}"]
    lines += [f"{r[0]:4d} {r[2]:8.3g} {r[3]:>9} {r[4]:8d} {r[5]:8d} {r[6]:7d} {r[7]:7d}" for r in rows]
    (fd / "summary.txt").write_text("\n".join(lines) + "\n")

    # 10. the same-rank, |cos| > 0.9 column of that table, over fig:vgap's zones
    def tick_step(K):
        return 10 if K <= 60 else 50 if K <= 300 else 200

    def match_bars(ax_, counts, K, title, xlabel=True, top_axis=True):
        """Matched-PC counts per step over fig:vgap's zones; top_axis adds the noise level t and the zone labels."""
        st.style(ax_, grid=False)
        ax_.yaxis.grid(True, color=st.GRID, lw=0.5)
        zones(ax_)
        ax_.bar(ks, counts, width=0.72, color=st.COND, zorder=3)
        ax_.set_xlim(0.5, S + 0.5)
        ax_.set_ylim(0, (1.24 if top_axis else 1.06) * K)
        ax_.set_yticks(np.arange(0, K + 1, tick_step(K)) if K >= 30 else np.linspace(0, K, 5).round().astype(int))
        ax_.set_xticks([1, 10, first, 20, 30, last, S])
        if xlabel:
            ax_.set_xlabel("sampling step  (1 = pure noise)")
        ax_.set_ylabel(f"matched PCs (of {K})")
        if top_axis:
            t_axis(ax_)
        head(ax_, title)

    def t_axis(ax_, y=0.984):
        """The noise level t along the top, and fig:vgap's zone names at height y (a fraction of the axes)."""
        from matplotlib.transforms import blended_transform_factory
        top = ax_.secondary_xaxis("top")
        tk = [1, first, 25, last, S]
        top.set_xticks(tk)
        top.set_xticklabels(["$\\infty$" if not np.isfinite(tlev[q - 1]) else f"{tlev[q - 1]:.2g}" for q in tk])
        top.set_xlabel("noise level $t$", color=st.MUTED)
        top.tick_params(colors=st.MUTED, labelcolor=st.MUTED, width=0.6, length=2.5)
        top.spines["top"].set_color(st.MUTED)
        ax_._has_top = True
        kw = dict(fontsize=7.5, ha="center", va="top", linespacing=1.1,
                  transform=blended_transform_factory(ax_.transData, ax_.transAxes))
        ax_.text(first / 2, y, "guidance harmful\n(high noise)", color="#b04a1e", **kw)
        ax_.text((first + last) / 2, y, f"guidance interval $t\\in({SIG_LO:g}, {SIG_HI:g}]$\n"
                 "(Kynkäänniemi et al., 2024)", color=st.INK, **kw)
        ax_.text((last + S + 1) / 2, y, "largely\nunneeded", color=st.MUTED, **kw)

    fig, ax_ = plt.subplots(figsize=(7.2, 3.2))
    match_bars(ax_, [r[4] for r in rows], C, r"PCs with $|\cos(v_i(\Sigma_c),\, v_i(\Sigma_u))| > 0.9$, same rank i")
    fig.suptitle(sub, y=1.18, fontsize=8, color=st.MUTED)
    save(fig, "pc_match")

    # 11-13. full-dimensional PCs of one covariance against another's (stages full and x0), per pair and K
    def full_figs(F, K, la, lb, ma, mb, sa, sb, name_cos, name_match, note=""):
        """Figures and tables for the top-K PCs of Cov(la) against those of Cov(lb), F as pair_pca returns it."""
        mC = min(C, K)
        fcos, fO, l_a, l_b = F["cos"], F["overlap"], F["lam_a"], F["lam_b"]
        selfc = np.minimum(F["selfcos_a"], F["selfcos_b"])
        Kh = int(np.isfinite(selfc[0]).sum())
        subf = sub + f"; full-dimensional PCs (57,600 dims) of Cov({la}) against Cov({lb})" + note
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1, 1], wspace=0.42))
        for i in range(mC):
            ax[0].plot(ks, fcos[:, i], color=pcol[i], lw=1.4, zorder=3 + C - i)
        ax[0].set_ylim(0, 1.03)
        ax[0].set_ylabel(rf"$|\cos(v_i({ma}),\, v_i({mb}))|$")
        st.style(ax[0])
        step_axis(ax[0])
        zones(ax[0], legend=True)
        pc_colorbar(fig, ax[0])
        head(ax[0], f"(a) full-dimensional PC i of Cov({la}) against PC i of Cov({lb}), top {mC}")
        yt = [1] + list(range(tick_step(K), K + 1, tick_step(K)))
        seq_bad = seq.with_extremes(bad=st.GRID)
        c_title = "(c) how well PC i is pinned down: |cos| between\n     two seed halves, the worse of the two fields"
        if Kh < K:
            c_title += f"\n     (grey: past the halves' own {Kh + 1} PCs)"
        for j, (M, tt) in enumerate([(fcos, f"(b) the top {K} as a heatmap (white dashes: the guidance interval)"),
                                     (selfc, c_title)]):
            im = ax[j + 1].imshow(np.ma.masked_invalid(M.T), aspect="auto", cmap=seq_bad, vmin=0, vmax=1,
                                  interpolation="nearest", extent=(0.5, S + 0.5, K + 0.5, 0.5))
            ax[j + 1].set_ylabel("PC rank i")
            ax[j + 1].set_yticks(yt)
            st.style(ax[j + 1], grid=False)
            step_axis(ax[j + 1], top=False)
            zones(ax[j + 1], shade=False)
            head(ax[j + 1], tt)
        cb = fig.colorbar(im, ax=ax[1:], fraction=0.03, pad=0.02)
        cb.set_label("|cos|")
        cb.outline.set_visible(False)
        fig.suptitle(subf, y=1.14, fontsize=9, color=st.MUTED)
        save(fig, name_cos)

        # per step and for the top m = 16 and K: matched PCs (same rank, and best match among the other's top m),
        # the mean cos^2 of the principal angles between the two top-m subspaces, the variance the top m carry, and
        # the variance carried by the same-rank PCs matched above 0.9 (a share of the total, the first field's)
        rows_f, cols = [], ("same_gt0.9", "same_gt0.8", "any_gt0.9", "any_gt0.8", "subspace_cos2", f"varfrac_{sa}",
                            f"varfrac_{sb}", f"varmatched_{sa}")
        for k in range(S):
            r = [int(ks[k]), float(s[k]), float(tlev[k]), str(zone[k])]
            for m in (mC, K):
                Om = fO[k, :m, :m].astype(np.float64)
                r += [int((fcos[k, :m] > .9).sum()), int((fcos[k, :m] > .8).sum()), int((Om.max(1) > .9).sum()),
                      int((Om.max(1) > .8).sum()), float((Om ** 2).sum() / m),
                      float(l_a[k, :m].sum() / l_a[k].sum()), float(l_b[k, :m].sum() / l_b[k].sum()),
                      float(l_a[k, :m][fcos[k, :m] > .9].sum() / l_a[k].sum())]
            rows_f.append(r)
        hdr = "step,s,t,zone," + ",".join(f"top{m}_{c}" for m in (mC, K) for c in cols)
        (fd / f"{name_match}.csv").write_text(hdr + "\n" + "\n".join(",".join(str(v) for v in r) for r in rows_f) + "\n")
        vlab = f"var {sa}/{sb}"
        lines.extend(["", f"Full-dimensional PCs (57,600 dims) of Cov({la}) matched to Cov({lb}){note}, same rank, |cos| "
                          f"above 0.9 / 0.8; sub = mean cos^2 of the principal angles between the top-m subspaces; var = "
                          f"share of the total variance in the top m ({sa} / {sb}); vm = share of {la}'s total variance in "
                          "the PCs matched above 0.9",
                      f"{'step':>4} {'t':>8} {'zone':>9} | top {mC}: {'>.9':>4} {'>.8':>4} {'sub':>5} {vlab:>11} {'vm':>5} | "
                      f"top {K}: {'>.9':>4} {'>.8':>4} {'sub':>5} {vlab:>11} {'vm':>5}"])
        lines.extend(f"{r[0]:4d} {r[2]:8.3g} {r[3]:>9} | {'':{len(str(mC)) + 5}}{r[4]:4d} {r[5]:4d} {r[8]:5.2f} "
                     f"{r[9]:5.2f}/{r[10]:5.2f} {r[11]:5.2f} | {'':{len(str(K)) + 5}}{r[12]:4d} {r[13]:4d} {r[16]:5.2f} "
                     f"{r[17]:5.2f}/{r[18]:5.2f} {r[19]:5.2f}" for r in rows_f)
        (fd / "summary.txt").write_text("\n".join(lines) + "\n")

        fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.9))
        match_bars(ax[0], [r[4] for r in rows_f], mC, f"(a) top {mC} PCs of Cov(${la}$) and Cov(${lb}$) with "
                   "$|\\cos| > 0.9$, same rank", xlabel=False)
        match_bars(ax[1], [r[12] for r in rows_f], K, f"(b) top {K} PCs of Cov(${la}$) and Cov(${lb}$) with "
                   "$|\\cos| > 0.9$, same rank")
        fig.tight_layout(h_pad=4.5)
        fig.suptitle(subf, y=1.04, fontsize=8, color=st.MUTED)
        save(fig, name_match)
        return fcos                                                  # same-rank |cos| [S, K]

    by_k = lambda f: int(f.stem.split("-k")[1])                     # noqa: E731
    counts = {}                                                     # (pair, K) -> same-rank |cos| [S, K]
    for fz in sorted(out.glob("pca_full-k*.npz"), key=by_k):
        Fz = np.load(fz)
        K = int(Fz["K"])
        F = dict(cos=Fz["cos"], overlap=Fz["overlap"], lam_a=Fz["lam_c"], lam_b=Fz["lam_u"],
                 selfcos_a=Fz["selfcos_c"], selfcos_b=Fz["selfcos_u"])
        counts["DcDu", K] = full_figs(F, K, "D_c", "D_u", r"\Sigma_c", r"\Sigma_u", "c", "u", f"pca_cos_full-k{K}",
                                      f"pc_match_full-k{K}")
    for key, la, ma, sa, lb, mb, sb in X0_PAIRS:
        for fz in sorted(out.glob(f"pca_x0-{key}-k*.npz"), key=by_k):
            Fz = np.load(fz)
            K = int(Fz["K"])
            counts[key, K] = full_figs(Fz, K, la, lb, ma, mb, sa, sb, f"pca_cos_x0-{key}-k{K}",
                                       f"pc_match_x0-{key}-k{K}", note=" (x_0: the final latents of these trajectories)")

    # 14. the three pairs in one figure, stacked on one step axis: D_c vs D_u, D_c vs x_0, D_u vs x_0, at several
    # cutoffs m. PC i's same-rank cosine does not depend on how many PCs were computed, so every m reads the largest K.
    trio = (("DcDu", "D_c", "D_u"), ("Dc", "D_c", "x_0"), ("Du", "D_u", "x_0"))
    Ks = sorted({K for (p, K) in counts if all((q, K) in counts for q, _, _ in trio)})
    if Ks:
        K = max(Ks)
        for m in sorted({m for m in (C, 20, 50, 100, 200) if m < K} | {K}):
            fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
            for n_, ((p, la, lb), a_) in enumerate(zip(trio, ax)):
                match_bars(a_, (counts[p, K][:, :m] > .9).sum(1), m, f"({'abc'[n_]}) Cov(${la}$) against "
                           f"Cov(${lb}$): top {m} full-dimensional PCs with $|\\cos| > 0.9$, same rank",
                           xlabel=n_ == 2, top_axis=n_ == 0)
            fig.tight_layout(h_pad=2.2)
            fig.suptitle(sub + f"; x_0: the final latents of these trajectories", y=1.03, fontsize=8, color=st.MUTED)
            save(fig, f"pc_match_pairs-top{m}")

    # 15. alignment without matching PCs one by one (stage align): the subspace overlap S_k, the covariance cosine
    # S_cov and the scale a* for the three pairs; then a control at 500 seeds, on shared and on disjoint seed halves
    PAIRS = (("DcDu", "D_c", "D_u", "#2a78d6"), ("Dc", "D_c", "x_0", "#eb6834"), ("Du", "D_u", "x_0", "#1baf7a"))
    unit = (0, 1.22)

    def frame(ax_, ylim=unit, top=True, xlabel=True):
        st.style(ax_)
        zones(ax_)
        ax_.set_xlim(0.5, S + 0.5)
        ax_.set_xticks([1, 10, first, 20, 30, last, S])
        if ylim == unit:
            ax_.set_ylim(*unit)
            ax_.set_yticks([0, 0.25, 0.5, 0.75, 1])
        if xlabel:
            ax_.set_xlabel("sampling step  (1 = pure noise)")
        if top:
            t_axis(ax_)

    def legend(ax_, loc="lower right"):
        hs = [h for h in ax_.get_lines() if not h.get_label().startswith("_")]     # the curves, not the zones
        ax_.legend(handles=hs, loc=loc, frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.5)

    if (out / "align.npz").exists():
        L = np.load(out / "align.npz")
        sub_a = sub + "; full-dimensional covariances (57,600 dims) over all seeds; x_0: the final latents"
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True, gridspec_kw=dict(wspace=0.1))
        kmax, khalf = L["DcDu_paired_sub"].shape[1], L["DcDu_cross_sub"].shape[1]     # small runs have fewer PCs
        for a_, k, n_ in zip(ax, [min(k, kmax) for k in (10, 50, 100)], "abc"):
            for p, la, lb, col in PAIRS:
                a_.plot(ks, L[f"{p}_paired_sub"][:, k - 1], color=col, lw=1.8, label=f"${la}$ vs ${lb}$", zorder=4)
            frame(a_)
            head(a_, f"({n_}) top {k}: $S_{{{k}}} = \\|U_a^\\top U_b\\|_F^2 / {k}$  (random subspaces: "
                     f"{k / (C * P):.1g})")
        ax[0].set_ylabel("subspace overlap $S_k$\n(mean cos$^2$ of the principal angles)")
        legend(ax[0])
        fig.suptitle(sub_a, y=1.12, fontsize=9, color=st.MUTED)
        save(fig, "align_subspace")

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(wspace=0.3))
        for p, la, lb, col in PAIRS:
            lab, sc_ = f"${la}$ vs ${lb}$", L[f"{p}_paired_scov"]
            ax[0].plot(ks, sc_, color=col, lw=1.8, label=lab, zorder=4)
            res_ = np.sqrt(np.clip(1 - sc_ ** 2, 0, None))
            ax[1].plot(ks, np.where(res_ > 1e-6, res_, np.nan), color=col, lw=1.8, label=lab, zorder=4)
            ax[2].plot(ks, L[f"{p}_paired_a"], color=col, lw=1.8, label=lab, zorder=4)
            ax[2].plot(ks, L[f"{p}_tr"], color=col, lw=1.2, ls=":", zorder=4)
        frame(ax[0])
        frame(ax[1], ylim=None)
        frame(ax[2], ylim=None)
        ax[1].set_yscale("log")
        ax[2].set_yscale("log")
        ax[0].set_ylabel(r"$S_\mathrm{cov}$")
        ax[1].set_ylabel(r"$\|\Sigma_a - a^*\Sigma_b\|_F \,/\, \|\Sigma_a\|_F$")
        ax[2].set_ylabel(r"$a^*$ (solid),  $\mathrm{tr}\,\Sigma_a / \mathrm{tr}\,\Sigma_b$ (dotted)")
        head(ax[0], r"(a) covariance cosine $S_\mathrm{cov} = \mathrm{tr}(\Sigma_a\Sigma_b) \,/\, "
                    r"(\|\Sigma_a\|_F \|\Sigma_b\|_F)$")
        head(ax[1], r"(b) left after the best global scale: $\sqrt{1 - S_\mathrm{cov}^2}$")
        head(ax[2], r"(c) that scale, $a^* = \mathrm{tr}(\Sigma_a\Sigma_b) \,/\, \|\Sigma_b\|_F^2$")
        legend(ax[0])
        fig.suptitle(sub_a, y=1.12, fontsize=9, color=st.MUTED)
        save(fig, "align_cov")

        fig, ax = plt.subplots(2, 3, figsize=(16, 8.6), sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.42))
        for j, (p, la, lb, col) in enumerate(PAIRS):
            k50 = min(50, khalf)
            for i, (q, name) in enumerate((("sub", f"$S_{{{k50}}}$"), ("scov", r"$S_\mathrm{cov}$"))):
                a_ = ax[i, j]
                get = lambda c: L[f"{p}_{c}_{q}"][:, k50 - 1] if q == "sub" else L[f"{p}_{c}_{q}"]   # noqa: E731
                a_.plot(ks, get("paired_half"), color=col, lw=1.8, label=f"${la}$ vs ${lb}$, the same seeds", zorder=4)
                a_.plot(ks, get("cross"), color=col, lw=1.8, ls="--", label=f"${la}$ vs ${lb}$, disjoint seeds",
                        zorder=4)
                a_.plot(ks, get("self_a"), color=st.MUTED, lw=1.3, ls=":", label=f"${la}$ vs ${la}$, disjoint seeds",
                        zorder=4)
                a_.plot(ks, get("self_b"), color=st.MUTED, lw=1.3, ls="-.", label=f"${lb}$ vs ${lb}$, disjoint seeds",
                        zorder=4)
                frame(a_, top=i == 0, xlabel=i == 1)
                head(a_, f"({'abcdef'[3 * i + j]}) {name}, ${la}$ vs ${lb}$")
                if j == 0:
                    a_.set_ylabel(name)
                if i == 0:
                    legend(a_)
        fig.suptitle(sub + f"; every curve from {meta['n'] // 2}-seed halves (even and odd batches): two fields on the same seeds, "
                     "on disjoint seeds, and one field's two halves (the estimation-noise ceiling)",
                     y=0.99, fontsize=9, color=st.MUTED)
        save(fig, "align_control")

        cols = [(f"{p}_{c}", L[f"{p}_{key}"] if m == 0 else L[f"{p}_{key}"][:, min(m, L[f"{p}_{key}"].shape[1]) - 1])
                for p, *_ in PAIRS
                for c, key, m in (("S10", "paired_sub", 10), ("S50", "paired_sub", 50), ("S100", "paired_sub", 100),
                                  ("Scov", "paired_scov", 0), ("a", "paired_a", 0), ("tr", "tr", 0),
                                  ("half_S50", "paired_half_sub", 50), ("cross_S50", "cross_sub", 50),
                                  ("selfa_S50", "self_a_sub", 50), ("selfb_S50", "self_b_sub", 50),
                                  ("half_Scov", "paired_half_scov", 0), ("cross_Scov", "cross_scov", 0),
                                  ("selfa_Scov", "self_a_scov", 0), ("selfb_Scov", "self_b_scov", 0))]
        hdr = "step,s,t,zone," + ",".join(c for c, _ in cols)
        body = [",".join([str(ks[k]), f"{s[k]:.6g}", f"{tlev[k]:.6g}", str(zone[k])] +
                         [f"{v[k]:.6g}" for _, v in cols]) for k in range(S)]
        (fd / "align.csv").write_text(hdr + "\n" + "\n".join(body) + "\n")

    # 16. the top-K ellipsoids of two covariances at a few steps, cut by one principal plane (principal_plane), under
    # the S_K curve they come from: D_c against D_u and against x_0 (stage align), D_w against x_0 (stage alignw)
    ell_steps = [int(q) for q in str(getattr(a, "ell_steps", "2,15,37")).split(",")]
    zname = lambda q: ("guidance harmful" if q < first else "guidance interval" if q <= last   # noqa: E731
                       else "largely unneeded")
    VIO = "#6250d6"                                         # D_w: the next free categorical hue, not w = 1's blue
    how = ("Below, per pair and at each marked step: the top-$K$ ellipsoids of the two covariances in one principal "
           "plane. $p$ and $q$ are one direction from each top-$K$\nsubspace, at their principal angle; each ellipse is "
           "the exact 1-sigma section of its covariance, its long axis exactly on its own direction. Drawn: the pair "
           "whose cos$^2$ is\nnearest $S_K$. Each panel has its own scale (latent units). ")

    def draw_ellipses(ae, PL, th, col, labs, R, lw=(1.4, 1.8), fs=8.5, alw=0.7, angle=True):
        """The two 1-sigma ellipses of PL [2, 2, 2] (reference in gray, then the other in col), each with its long
        axis dashed out to R, and (angle) the angle th between them as a labelled arc."""
        from matplotlib.patches import Arc
        for C2, edge, fill, alpha, lw_, lab in ((PL[0], st.MUTED, st.PHOTO, 0.28, lw[0], labs[0]),
                                                (PL[1], col, col, 0.14, lw[1], labs[1])):
            ev, V = np.linalg.eigh(C2)
            phi = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            ae.add_patch(Ellipse((0, 0), 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0]), angle=phi, facecolor=fill,
                                 alpha=alpha, lw=0, zorder=2))
            ae.add_patch(Ellipse((0, 0), 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0]), angle=phi, fill=False,
                                 edgecolor=edge, lw=lw_, zorder=3, label=lab))
            u = V[:, 1] * np.sign(V[1, 1] if abs(V[1, 1]) > 1e-12 else V[0, 1]) * R
            ae.plot([-u[0], u[0]], [-u[1], u[1]], color=edge, lw=alw, ls=(0, (4, 2)), zorder=2)
        if not angle:
            return
        ae.add_patch(Arc((0, 0), 0.62 * R, 0.62 * R, theta1=0, theta2=th, color=st.INK, lw=0.9 * alw / 0.7,
                         zorder=4))
        at = np.radians(th + 6)                             # just past the arc's end, clear of both axes
        ae.text(0.36 * R * np.cos(at), 0.36 * R * np.sin(at), f"{th:.0f}°" if th >= 1 else f"{th:.1f}°",
                fontsize=fs, color=st.INK, ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))

    def ellipse_panel(ae, L_, p, la, lb, col, K, q, note="", head_=None):
        """Pair p's top-K ellipsoids at step q in its principal plane: lb's in gray along the first axis, la's in col
        at the drawn principal angle, each the exact 1-sigma section of its covariance, at the panel's own scale.
        Titled with the step, its zone, S_K and the drawn angle (after head_, if given). Returns (all angles, drawn)."""
        PL, angs = L_[f"{p}_plane_k{K}"][q - 1], L_[f"{p}_ang_k{K}"][q - 1]
        th = angs[int(L_[f"{p}_pick_k{K}"][q - 1])]
        R = 1.1 * max(np.sqrt(np.linalg.eigvalsh(PL[j]).max()) for j in (0, 1))
        st.style(ae, grid=False)
        draw_ellipses(ae, PL, th, col, (f"${lb}$", f"${la}$" + note), R)
        ae.set_xlim(-R, R)
        ae.set_ylim(-R, R)
        ae.set_aspect("equal")
        ae.set_xlabel(f"along $p$, in ${lb}$'s top {K}", labelpad=1)
        tq = ("\\infty" if not np.isfinite(tlev[q - 1]) else f"{tlev[q - 1]:.0f}" if tlev[q - 1] >= 10
              else f"{tlev[q - 1]:.2g}")
        Sk = np.mean(np.cos(np.radians(angs)) ** 2)
        ae.text(0, 1.03, (head_ or "") + f"step {q} ($t = {tq}$), {zname(q)}\n"
                f"$S_{{{K}}} = {Sk:.{3 if Sk > 0.99 else 2}f}$; drawn: the pair at {th:.0f}°",
                transform=ae.transAxes, fontsize=8.5, color=st.INK, va="bottom")
        return angs, th

    def ellipse_figure(rows, K, steps, name, sup, hist=True):
        """S_K against the step for each row's pair, the steps marked; then per pair (one row each) the top-K
        ellipsoids of its two covariances at those steps in one principal plane, and (hist) all K principal angles below.
        rows: (npz, key, field drawn at the angle, reference field, colour[, label suffix]); the npz holds
        <key>_paired_sub and the principal planes (<key>_{ang,pick,plane}_k<K>)."""
        block = (3, 0.95, 0.62) if hist else (3, 0.62)      # per pair: ellipses, [histograms,] a gap
        hr = [2.3, 0.62] + [h for _ in rows for h in block][:-1]
        fig = plt.figure(figsize=(3.5 * len(steps) + 0.4, 3.6 + (4.7 if hist else 3.75) * len(rows)))
        gs = fig.add_gridspec(len(hr), len(steps), height_ratios=hr, hspace=0.38, wspace=0.34)
        ac = fig.add_subplot(gs[0, :])
        for L_, p, la, lb, col, *note in rows:
            y = L_[f"{p}_paired_sub"][:, K - 1]
            ac.plot(ks, y, color=col, lw=1.8, label=f"${la}$ vs ${lb}$" + "".join(note), zorder=4)
            for q in steps:
                ac.plot([q], [y[q - 1]], "o", ms=7, mfc="white", mec=col, mew=1.5, zorder=6, label="_")
        for q in steps:                                     # under the lowest marker at that step
            y_ = min(L_[f"{p}_paired_sub"][q - 1, K - 1] for L_, p, *_ in rows)
            ac.annotate(f"step {q}", (q, y_), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8,
                        color=st.INK, zorder=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        frame(ac)
        ac.set_ylabel(f"$S_{{{K}}}$")
        ac.text(0, 1.36, f"(a) subspace overlap of the top-{K} PCs, $S_{{{K}}} = \\|U_a^\\top U_b\\|_F^2 / {K}$: the mean "
                f"cos$^2$ of the {K} principal angles (random subspaces: {K / (C * P):.1g})", transform=ac.transAxes,
                fontsize=9.5, color=st.INK, va="bottom")
        legend(ac)
        for r, (L_, p, la, lb, col, *note) in enumerate(rows):
            hmax = max(np.histogram(L_[f"{p}_ang_k{K}"][q - 1], bins=np.arange(0, 91, 5))[0].max() for q in steps)
            for cix, q in enumerate(steps):
                ae = fig.add_subplot(gs[2 + len(block) * r, cix])
                angs, th = ellipse_panel(ae, L_, p, la, lb, col, K, q, "".join(note))
                if cix == 0:
                    ae.set_ylabel(f"toward $q$, in ${la}$'s top {K}", labelpad=1)
                    ae.text(0, 1.27, f"({'bcdefg'[r]}) Cov(${la}$) against Cov(${lb}$)" + "".join(note) +
                            f": the top-{K} ellipsoids",
                            transform=ae.transAxes, fontsize=9.5, color=st.INK, va="bottom")
                    ae.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", fontsize=8)
                if not hist:
                    continue
                ah = fig.add_subplot(gs[3 + len(block) * r, cix])
                st.style(ah, grid=False)
                ah.yaxis.grid(True, color=st.GRID, lw=0.5)
                ah.hist(angs, bins=np.arange(0, 91, 5), color=col, rwidth=0.82, zorder=3)
                ah.axvline(th, color=st.INK, lw=0.9, ls=(0, (2, 1.5)), zorder=4)
                ah.set_xlim(0, 90)
                ah.set_xticks([0, 15, 30, 45, 60, 75, 90])
                ah.set_ylim(0, 1.15 * hmax)
                ah.set_xlabel(f"all {K} principal angles (degrees); median {np.median(angs):.0f}°", labelpad=1)
                if cix == 0:
                    ah.set_ylabel("count")
        fig.suptitle(sup, y=0.997, fontsize=8, color=st.MUTED)
        fig.subplots_adjust(top=1 - 1.45 / fig.get_figheight())
        save(fig, name)

    def overview(L, K, steps, name, column=False):
        """One text-width figure: (a) fig:vgap's curve over (b) S_K for D_c vs D_u and D_c vs x_0, on one step axis
        with the steps marked; right, the top-K ellipses at those steps (columns) on one scale, D_c against D_u above
        and against x_0 below. column: the same at one column's width, the ellipses under (b), laid out in inches."""
        import matplotlib
        coco, ref = vgap_curve()
        run = vgap_run(out)
        gap = run["total"] if run is not None else coco             # this run's seeds when its sums are here
        pairs = (("DcDu", "D_c", "D_u", "#2a78d6"), ("Dc", "D_c", "x_0", "#eb6834"))
        tq = lambda q: ("\\infty" if not np.isfinite(tlev[q - 1]) else f"{tlev[q - 1]:.0f}"   # noqa: E731
                        if tlev[q - 1] >= 10 else f"{tlev[q - 1]:.2g}")
        rc = {"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
              "legend.fontsize": 6.5, "axes.titlesize": 7.5}
        with matplotlib.rc_context(rc):
            ey = 0.6 if column else 0.7                     # the ellipse panels' height over width: they are flat
            if column:
                W = (6.875 - 0.3125) / 2                    # CVPR \\columnwidth, included at 100%
                Lm, Rm, T0, ha, hb, gab, gmid, ge, B = 0.47, 0.11, 0.33, 0.74, 0.86, 0.04, 0.44, 0.07, 0.31
                we = (W - Lm - Rm - (len(steps) - 1) * ge) / len(steps)
                H = T0 + ha + gab + hb + gmid + len(pairs) * ey * we + (len(pairs) - 1) * ge + B
                fig = plt.figure(figsize=(W, H))

                def rect(x, ytop, w, h):                    # inches from the top left -> figure fractions
                    return [x / W, (H - ytop - h) / H, w / W, h / H]
                av = fig.add_axes(rect(Lm, T0, W - Lm - Rm, ha))
                ab = fig.add_axes(rect(Lm, T0 + ha + gab, W - Lm - Rm, hb), sharex=av)
                new_ell = lambda r, c, **kw: fig.add_axes(rect(Lm + c * (we + ge), T0 + ha + gab + hb + gmid   # noqa: E731
                                                               + r * (ey * we + ge), we, ey * we), **kw)
            else:
                fig = plt.figure(figsize=(6.875, 3.0))      # CVPR \\textwidth: included at 100%, fonts at size
                outer = fig.add_gridspec(1, 2, width_ratios=[1.22, 1], left=0.072, right=0.998, bottom=0.125,
                                         top=0.885, wspace=0.15)
                gl = outer[0].subgridspec(2, 1, hspace=0.07)
                gr = outer[1].subgridspec(2, len(steps), hspace=0.05, wspace=0.05)
                av = fig.add_subplot(gl[0])
                ab = fig.add_subplot(gl[1], sharex=av)
                new_ell = lambda r, c, **kw: fig.add_subplot(gr[r, c], **kw)   # noqa: E731
            for ax_ in (av, ab):
                st.style(ax_)
                ax_.axvspan(0.5, first - 0.5, color=st.ACC, alpha=0.09, lw=0, zorder=0)
                ax_.axvspan(last + 0.5, S + 0.5, color=st.MUTED, alpha=0.12, lw=0, zorder=0)
                for xb in (first - 0.5, last + 0.5):
                    ax_.axvline(xb, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
                for q in steps:
                    ax_.axvline(q, color=st.INK, lw=0.5, ls=":", zorder=1)
            av.plot(ks, gap, color=st.UNCOND, lw=1.2, zorder=3)
            av.scatter(ks, gap, s=4, color=st.UNCOND, lw=0, zorder=4)
            av.plot(steps, gap[np.array(steps) - 1], "o", ms=4.5, mfc="white", mec=st.UNCOND, mew=1.1, zorder=5)
            av.set_yscale("log")
            av.set_ylim(gap.min() / 2, gap.max() * 9)
            av.set_yticks([1e-3, 1e-2, 1e-1, 1])                # every decade labelled, also when short
            av.set_ylabel(r"$\mathbb{E}\,\|v_c - v_u\|^2$")
            if ref is not None and run is None:             # E||v*||^2 is the COCO pass's, not this run's
                right = av.secondary_yaxis("right", functions=(lambda y: 100 * y / ref, lambda q: q * ref / 100))
                right.set_ylabel(r"% of $\mathbb{E}\,\|v^\star\|^2$")
            av.tick_params(labelbottom=False)
            top = av.secondary_xaxis("top")
            tk = [1, first, 25, last, S]
            top.set_xticks(tk)
            top.set_xticklabels([f"${tq(q)}$" for q in tk])
            top.set_xlabel("noise level $t$", labelpad=2, color=st.MUTED)
            top.tick_params(colors=st.MUTED, labelcolor=st.MUTED, width=0.6, length=2.5, pad=1.5)
            top.spines["top"].set_color(st.MUTED)
            from matplotlib.transforms import blended_transform_factory
            kw = dict(fontsize=5.8 if column else 6.3, ha="center", va="top", linespacing=1.0,
                      transform=blended_transform_factory(av.transData, av.transAxes))
            av.text(first / 2 + 1, 0.96, "guidance\nharmful", color="#b04a1e", **kw)
            av.text((first + last) / 2, 0.96, "useful guidance interval", color=st.INK, **kw)
            av.text((last + S + 1) / 2, 0.96, "un-\nneces-\nsary" if column else "unneces-\nsary",   # fits 4 steps
                    color=st.MUTED, **kw)
            av.text(0.015, 0.05, "(a)", transform=av.transAxes, fontsize=7.5, color=st.INK)

            for p, la, lb, col in pairs:
                y = L[f"{p}_paired_sub"][:, K - 1]
                ab.plot(ks, y, color=col, lw=1.3, label=f"${la}$ vs ${lb}$", zorder=4)
                ab.plot(steps, y[np.array(steps) - 1], "o", ms=4.5, mfc="white", mec=col, mew=1.1, zorder=5,
                        label="_")
            ab.axhline(1, color=st.INK, lw=0.8, ls=(0, (4, 2)), zorder=2)          # saturation: the same subspace
            ab.text(1.3, 1.006, "1: the same directions", fontsize=6.3, color=st.INK, va="bottom",
                    zorder=6, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
            ab.set_ylim(0.6, 1.05)
            ab.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
            ab.set_xlim(0.5, S + 0.5)
            ab.set_xticks(sorted({1, *steps, first, last, S}))
            ab.set_xlabel("sampling step", labelpad=1.5)
            ab.set_ylabel("direction alignment")           # S_K, the mean cos^2 of the K principal angles
            ab.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", handlelength=1.6,
                      borderpad=0.3, labelspacing=0.25)
            ab.text(0.015, 0.035, "(b)", transform=ab.transAxes, fontsize=7.5, color=st.INK)

            R = 1.06 * max(np.sqrt(np.linalg.eigvalsh(L[f"{p}_plane_k{K}"][q - 1, j]).max())
                           for p, *_ in pairs for q in steps for j in (0, 1))
            e0, eax = None, []
            for r, (p, la, lb, col) in enumerate(pairs):
                for c, q in enumerate(steps):
                    ae = new_ell(r, c, sharex=e0, sharey=e0)
                    eax.append(ae)
                    e0 = e0 or ae
                    angs = L[f"{p}_ang_k{K}"][q - 1]
                    th = angs[int(L[f"{p}_pick_k{K}"][q - 1])]
                    st.style(ae, grid=False)
                    draw_ellipses(ae, L[f"{p}_plane_k{K}"][q - 1], th, col, (f"${lb}$", f"${la}$"), R,
                                  lw=(1.0, 1.25), fs=6.5, alw=0.5, angle=False)
                    ae.set_xlim(-R, R)
                    ae.set_ylim(-ey * R, ey * R)
                    ae.set_aspect("equal", adjustable="box")
                    ae.tick_params(labelbottom=r == len(pairs) - 1, labelleft=c == 0)
                    Sk = np.mean(np.cos(np.radians(angs)) ** 2)
                    ae.text(0.04, 0.95, f"({'cdefgh'[r * len(steps) + c]}) alignment {Sk:.{3 if Sk > 0.99 else 2}f}",
                            transform=ae.transAxes, fontsize=6.5, color=st.INK, va="top")
                    if r == 0:
                        ae.set_title(f"step {q}", pad=3)
                    if c == len(steps) - 1:
                        ae.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none",
                                  handlelength=1.3, borderpad=0.25, labelspacing=0.2, handletextpad=0.4)
            pos = [a_.get_position() for a_ in eax]
            x0, x1 = min(b.x0 for b in pos), max(b.x1 for b in pos)
            y0, Hf = min(b.y0 for b in pos), fig.get_figheight()
            fig.canvas.draw()
            if column:                                      # under the block
                yl, va = y0 - 0.17 / Hf, "top"
            else:                                           # level with the step axis's label
                yl, va = ab.xaxis.label.get_window_extent().y0 / fig.bbox.height, "bottom"
            fig.text((x0 + x1) / 2, yl, f"along $p$, in the gray field's top {K}", ha="center", va=va,
                     fontsize=7.5, color=st.INK)
            fig.savefig(fd / f"{name}.pdf")                 # no cropping: exactly the page's text width
            fig.savefig(fd / f"{name}.png", dpi=300)
            plt.close(fig)

    if (out / "align.npz").exists() and f"Dc_plane_k{ELL_K[0]}" in np.load(out / "align.npz"):
        L = np.load(out / "align.npz")
        K = ELL_K[0]
        rows = [(L, "DcDu", "D_c", "D_u", "#2a78d6"), (L, "Dc", "D_c", "x_0", "#eb6834")]
        ellipse_figure(rows, K, ell_steps, f"align_ellipses-top{K}",
                       f"Top: how far the top-{K} PC subspaces of two covariances agree, step by step. " + how +
                       f'SD 3.5 Medium, $w = 1$, {meta["n"]} seeds, prompt "{meta["prompt"]}"; $x_0$: the final latents; '
                       "full-dimensional covariances (57,600 dims) over all seeds.", hist=False)
        ov = [int(q) for q in str(getattr(a, "overview_steps", "5,25")).split(",")]
        overview(L, K, ov, f"align_overview-top{K}")
        overview(L, K, ov, f"align_overview-col-top{K}", column=True)

    # 17. D_w = w D_c + (1 - w) D_u against x_0 (stage alignw): S_k as in align_subspace, and its ellipsoids
    for fw in sorted(out.glob("align_w*.npz")):
        W = np.load(fw)
        wv, tag = float(W["w"]), fw.stem[len("align_"):]
        ref = np.load(out / "align.npz") if (out / "align.npz").exists() else None
        lab_w = f"$D_w$ vs $x_0$,  $w = {wv:g}$"
        sub_w = (f'SD 3.5 Medium, {meta["n"]} seeds, prompt "{meta["prompt"]}"; $D_w = w D_c + (1 - w) D_u$ '
                 f"evaluated on the unguided ($w = 1$) trajectories; $x_0$: their final latents")

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True, gridspec_kw=dict(wspace=0.1))
        kmax = W["Dw_paired_sub"].shape[1]
        for a_, k, n_ in zip(ax, [min(k, kmax) for k in (10, 50, 100)], "abc"):
            a_.plot(ks, W["Dw_paired_sub"][:, k - 1], color=VIO, lw=2.0, label=lab_w, zorder=5)
            if ref is not None:
                for p, la, lb, col in PAIRS[1:]:
                    a_.plot(ks, ref[f"{p}_paired_sub"][:, k - 1], color=col, lw=1.3, label=f"${la}$ vs ${lb}$  "
                            f"($w = {1 if p == 'Dc' else 0}$)", zorder=4)
            if k in ELL_K:                                  # the steps drawn as ellipses
                for q in ell_steps:
                    y_ = W["Dw_paired_sub"][q - 1, k - 1]
                    a_.plot([q], [y_], "o", ms=7.5, mfc="white", mec=VIO, mew=1.5, zorder=6, label="_")
                    a_.annotate(f"step {q}", (q, y_), xytext=(0, -13), textcoords="offset points", ha="center",
                                fontsize=7.5, color=st.INK, zorder=7)
            frame(a_)
            head(a_, f"({n_}) top {k}: $S_{{{k}}} = \\|U_a^\\top U_b\\|_F^2 / {k}$  (random subspaces: "
                     f"{k / (C * P):.1g})")
        ax[0].set_ylabel("subspace overlap $S_k$\n(mean cos$^2$ of the principal angles)")
        legend(ax[0])
        fig.suptitle(sub_w + "; full-dimensional covariances over all seeds", y=1.12, fontsize=9, color=st.MUTED)
        save(fig, f"align_subspace_{tag}")

        for K in ELL_K:
            ellipse_figure([(W, "Dw", "D_w", "x_0", VIO)], K, ell_steps, f"ellipses_{tag}-top{K}",
                           f"Top: how far the top-{K} PC subspaces of Cov($D_w$), $w = {wv:g}$, and Cov($x_0$) agree, step "
                           "by step. " + how + "The histograms give all $K$ principal angles. " + sub_w + ".")

        cols = [(c, W[f"Dw_{key}"] if m == 0 else W[f"Dw_{key}"][:, min(m, W[f"Dw_{key}"].shape[1]) - 1])
                for c, key, m in (("S10", "paired_sub", 10), ("S50", "paired_sub", 50), ("S100", "paired_sub", 100),
                                  ("Scov", "paired_scov", 0), ("a", "paired_a", 0), ("tr", "tr", 0),
                                  ("half_S50", "paired_half_sub", 50), ("cross_S50", "cross_sub", 50),
                                  ("selfw_S50", "self_a_sub", 50), ("selfx_S50", "self_b_sub", 50),
                                  ("half_Scov", "paired_half_scov", 0), ("cross_Scov", "cross_scov", 0),
                                  ("selfw_Scov", "self_a_scov", 0), ("selfx_Scov", "self_b_scov", 0))]
        for K in ELL_K:
            A_ = W[f"Dw_ang_k{K}"]
            cols += [(f"theta_drawn_{K}", A_[np.arange(S), W[f"Dw_pick_k{K}"]]), (f"theta_median_{K}", np.median(A_, 1)),
                     (f"theta_max_{K}", A_.max(1))]
        hdr = "step,s,t,zone," + ",".join(c for c, _ in cols)
        body = [",".join([str(ks[k]), f"{s[k]:.6g}", f"{tlev[k]:.6g}", str(zone[k])] +
                         [f"{v[k]:.6g}" for _, v in cols]) for k in range(S)]
        (fd / f"align_{tag}.csv").write_text(hdr + "\n" + "\n".join(body) + "\n")

    print(f"figures in {fd}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("work", "analyze", "full", "x0", "align", "alignw", "figures"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--n", type=int, default=1000, help="number of seeds (initial noises)")
    ap.add_argument("--batch", type=int, default=25, help="seeds per batch, the unit of work and of resuming")
    ap.add_argument("--k", type=int, default=50, help="full and x0 stages: how many full-dimensional PCs to compare")
    ap.add_argument("--w", type=float, default=4.5, help="alignw: the guidance weight of D_w")
    ap.add_argument("--ell-steps", default="2,15,37", help="figures: the steps drawn as ellipses")
    ap.add_argument("--overview-steps", default="5,25", help="figures: the steps of align_overview's ellipse columns")
    a = ap.parse_args()
    {"work": work, "analyze": analyze, "full": full, "x0": x0, "align": align, "alignw": alignw,
     "figures": figures}[a.stage](a)


if __name__ == "__main__":
    main()
