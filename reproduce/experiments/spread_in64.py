#!/usr/bin/env python
"""Fig 5(c) on ImageNet-64: how the spread of retrained denoisers' predictions at a fixed x_t shrinks with N.

The 2-D panel trains 1,024 MLPs per training-set size N, each on its own N points, and plots
tr Cov_models D(x_t) / tr Sigma(x_t) against N. Here every model is an unconditional EDM2 UNet trained with
EDM2's own loop (Adam, EDM2Loss, power-function EMA) on its own random subset of the 1,281,167 ImageNet-64
train images, and the inputs are held-out val images noised as x_t = x_0 + sigma eps, the same eps for every
model. Three curves per noise level:

    trained          tr Cov over models that differ in subset and seed
    seed only        tr Cov over models that share one subset and differ in seed (optimization noise alone)
    exact mean       the posterior mean of the N training points themselves, sum_i k_i x_i / sum_i k_i with
                     k_i = N(x_t; x_i, sigma^2 I), over `exact_subsets` random subsets (no training)

tr Sigma has no closed form on ImageNet. It is Tweedie's second moment, Cov(x_0 | x_t) = sigma^2 dD*/dx_t,
with NVIDIA's unconditional EDM2-XS (2,147 Mi images, EMA 0.050) standing in for D*, and the trace estimated
with Rademacher probes (Hutchinson): tr Sigma = sigma^2 E_v[v . J^T v].

    python reproduce/experiments/spread_in64.py prep    [--src /path/to/ILSVRC2012/train]   # CPU, once
    python reproduce/experiments/spread_in64.py train   --out DIR --task {task}              # GPU, one run per task
    python reproduce/experiments/spread_in64.py probe   --out DIR                            # GPU, every snapshot
    python reproduce/experiments/spread_in64.py ref     --out DIR                            # GPU, tr Sigma
    python reproduce/experiments/spread_in64.py exact   --out DIR                            # GPU, exact means
    python reproduce/experiments/spread_in64.py analyze --out DIR                            # CPU
    python reproduce/experiments/spread_in64.py runs    --out DIR                            # list the tasks

The grid lives in DIR/config.json, written from PILOT by the first stage that runs (edit it before training
to change the grid). Training is resumable: rerunning a task continues from its latest training state.
`duration` is only where training stops: EDM2's learning-rate schedule and power-function EMA never read it,
so a 16 Mi run is the first half of a 32 Mi one. To train longer, raise it in config.json and rerun the tasks.

DIR holds:
    config.json                   the grid and the training recipe
    subsets/n<N>-j<j>.npy         sorted train-row indices of subset j of size N
    probes.npz                    rows [P] (val), sigmas [L], x0 [P, 3, 64, 64], eps [L, P, 3, 64, 64] (float32)
    runs/<name>/                  EDM2 run directories; name = n<N>-j<j>-s<seed>; DONE holds the images trained
    preds/<name>/k<kimg>.npz      D(x_t) [L, P, 3, 64, 64] float32 for every EMA of the snapshot at <kimg>
    ref.npz                       tr_sigma, tr_se [L, P] (Hutchinson mean and standard error), mmse [L, P], d [L, P, ...]
    exact/n<N>.npz                tr_cov [L, P] over subsets, ess [L, P] (mean 1/sum w^2), d [K, L, P, ...] for the
                                  trained subsets
    analysis.npz, summary.txt, fig/
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

M_TRAIN = 1281167
RES = 64
TRAIN_NAME = "img64_train_u8.npy"           # uint8 [M, 64, 64, 3], ADM's centre crop, sorted synset listing
SALT = 20261112

PILOT = dict(
    ns=[4096, 65536],                       # training-set sizes
    k=4,                                    # models per N, each on its own subset j = 0..k-1 with seed j
    seed_only=[[65536, 0, [1, 2]]],         # [N, subset j, extra seeds]: the same data, other seeds
    channels=64, dropout=0.0, batch=512, duration=16 << 20, lr=0.010, decay=8000, rampup_Mimg=2,
    P_mean=-0.8, P_std=1.6,                 # edm2-img64-* noise distribution
    ema=[0.050, 0.100], snapshot=4 << 20, checkpoint=2 << 20, status=256 << 10, workers=4,
    sigmas=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0], probes=256, probe_seed=2,
    exact_subsets=32, hutchinson=16,
    ref="edm2-img64-xs-uncond-2147483-0.050",
)


# --- bookkeeping ---------------------------------------------------------------------------------------------
def imagenet_root():
    from vmg.data import imagenet
    return imagenet.root()


def train_array(per_class=0):
    """prep's array; per_class > 0: the smoke-sized one of that many images per class, kept apart from the full one."""
    return imagenet_root() / (TRAIN_NAME.replace(".npy", f"-k{per_class}.npy") if per_class else TRAIN_NAME)


def train_labels(per_class=0):
    p = train_array(per_class)
    return p.with_name(p.name.replace("_u8", "_labels"))


def atomic_save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def config(out):
    p = Path(out) / "config.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".config.tmp{os.getpid()}")
        tmp.write_text(json.dumps(PILOT, indent=1))
        try:
            os.link(tmp, p)                  # first writer wins; later ones read its file
        except FileExistsError:
            pass
        tmp.unlink()
    return json.loads(p.read_text())


def run_list(cfg):
    """Every training run as (name, N, subset j, seed), longest data first (all have the same duration)."""
    runs = []
    for n in sorted(cfg["ns"], reverse=True):
        runs += [(f"n{n}-j{j}-s{j}", n, j, j) for j in range(cfg["k"])]
        for n2, j, seeds in cfg["seed_only"]:
            if n2 == n:
                runs += [(f"n{n}-j{j}-s{s}", n, j, s) for s in seeds]
    return runs


def done_nimg(run_dir):
    """Training images a finished run reached (its DONE file holds them); 0 while unfinished."""
    p = Path(run_dir) / "DONE"
    if not p.exists():
        return 0
    first = (p.read_text().split() or [""])[0]
    return int(first) if first.isdigit() else 1 << 62


def subset(out, n, j):
    """Sorted train rows of subset j of size n: an independent uniform draw without replacement per (n, j)."""
    p = Path(out) / "subsets" / f"n{n}-j{j}.npy"
    if not p.exists():
        idx = np.sort(np.random.default_rng([n, j, SALT]).choice(M_TRAIN, n, replace=False)).astype(np.int64)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.stem}.tmp{os.getpid()}.npy")
        np.save(tmp, idx)
        os.replace(tmp, p)
    return p


def probes(out):
    """Held-out val images x0 and one fixed eps per (level, image)."""
    p = Path(out) / "probes.npz"
    if not p.exists():
        import torch
        from vmg.data import imagenet
        cfg = config(out)
        rows = np.asarray(imagenet.subset(imagenet.labels(64), cfg["probes"], seed=cfg["probe_seed"]))
        x0 = imagenet.pixels(rows).numpy()
        eps = np.stack([np.stack([torch.randn(x0.shape[1:], generator=torch.Generator().manual_seed(
            1_000_003 * l + int(r))).numpy() for r in rows]) for l in range(len(cfg["sigmas"]))])
        atomic_save(p, rows=rows, sigmas=np.asarray(cfg["sigmas"], np.float64), x0=x0, eps=eps)
    z = np.load(p)
    return {k: z[k] for k in z.files}


def edm2():
    from vmg.models import _edm
    _edm.edm_setup()
    return _edm


# --- data ----------------------------------------------------------------------------------------------------
class SubsetDataset:
    """Train images at the rows in `index` (an .npy), CHW uint8, no labels. The memmap opens lazily in each
    DataLoader worker so that pickling the dataset (spawn) never copies the array."""

    def __init__(self, path, index):
        self.path, self.index_path = str(path), str(index)
        self.index = np.load(index)
        self._arr = None

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        return dict(self.__dict__, _arr=None)

    def __getitem__(self, i):
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode="r")
        img = np.ascontiguousarray(self._arr[self.index[i]].transpose(2, 0, 1))
        return img, np.zeros([0], np.float32)


def prep(a):
    """The train JPEGs (one folder per synset) through ADM's centre crop at 64, into one uint8 array."""
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from vmg.data import imagenet
    assert a.src, "--src (or $VMG_IMAGENET_TRAIN): the ImageNet train JPEGs, one folder per synset"
    src = Path(a.src)
    syn = sorted(d.name for d in src.iterdir() if d.is_dir())
    assert len(syn) == 1000, len(syn)
    paths, lab = [], []
    for c, s in enumerate(syn):
        f = sorted(str(p) for p in (src / s).iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
        f = f[:a.per_class] if a.per_class else f
        paths += f
        lab += [c] * len(f)
    m = 1000 * a.per_class if a.per_class else M_TRAIN
    assert len(paths) == m, len(paths)
    root = imagenet_root()
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{train_array(a.per_class).name}.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(m, RES, RES, 3))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, im in enumerate(ex.map(partial(imagenet._crop, res=RES), paths, chunksize=256)):
            arr[i] = im
            if i % 100000 == 0:
                print(f"{i:,} / {m:,}  {time.time() - t0:.0f}s", flush=True)
    arr.flush()
    del arr
    os.replace(tmp, train_array(a.per_class))
    np.save(train_labels(a.per_class), np.asarray(lab, np.int16))
    print(f"wrote {train_array(a.per_class)}")


# --- training ------------------------------------------------------------------------------------------------
def train(a):
    cfg = config(a.out)
    runs = run_list(cfg)
    name, n, j, seed = runs[a.task]
    run_dir = Path(a.out) / "runs" / name
    if done_nimg(run_dir) >= cfg["duration"]:
        print(f"{name}: done")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "DONE").unlink(missing_ok=True)          # finished at a shorter duration: train on
    idx = subset(a.out, n, j)
    _edm = edm2()
    import torch
    import dnnlib
    from torch_utils import distributed as dist
    import training.training_loop as tl

    # a training state cut off mid-save would stop every resume: drop the latest if it does not load
    states = sorted(run_dir.glob("training-state-*.pt"))
    if states:
        try:
            torch.load(states[-1], map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"dropping unreadable {states[-1].name}: {e}")
            states[-1].unlink()

    torch.multiprocessing.set_start_method("spawn", force=True)
    dist.init()
    E = dnnlib.EasyDict
    c = E(
        dataset_kwargs=E(class_name="reproduce.experiments.spread_in64.SubsetDataset", path=str(train_array()),
                         index=str(idx)),
        encoder_kwargs=E(class_name="training.encoders.StandardRGBEncoder"),
        data_loader_kwargs=E(class_name="torch.utils.data.DataLoader", pin_memory=True,
                             num_workers=cfg["workers"], prefetch_factor=4, persistent_workers=True),
        network_kwargs=E(class_name="training.networks_edm2.Precond", model_channels=cfg["channels"],
                         dropout=cfg["dropout"], use_fp16=True),
        loss_kwargs=E(class_name="training.training_loop.EDM2Loss", P_mean=cfg["P_mean"], P_std=cfg["P_std"]),
        lr_kwargs=E(func_name="training.training_loop.learning_rate_schedule", ref_lr=cfg["lr"],
                    ref_batches=cfg["decay"], rampup_Mimg=cfg["rampup_Mimg"]),
        ema_kwargs=E(class_name="training.phema.PowerFunctionEMA", stds=cfg["ema"]),
        seed=seed, batch_size=cfg["batch"], total_nimg=cfg["duration"], status_nimg=cfg["status"],
        snapshot_nimg=cfg["snapshot"], checkpoint_nimg=cfg["checkpoint"], loss_scaling=1, cudnn_benchmark=True,
    )
    opts = run_dir / "training_options.json"
    if not opts.exists():
        opts.write_text(json.dumps(dict(c, name=name, n=n, subset=j, edm2=_edm.REPOS["edm2"][1]), indent=2))
    with open(run_dir / "gpu.txt", "a") as f:
        f.write(torch.cuda.get_device_name() + "\n")
    dnnlib.util.Logger(file_name=str(run_dir / "log.txt"), file_mode="a", should_flush=True)
    tl.training_loop(run_dir=str(run_dir), **c)

    # keep the last training state only (it is what a longer duration resumes from); the snapshots stay
    for p in sorted(run_dir.glob("training-state-*.pt"))[:-1]:
        p.unlink()
    (run_dir / "DONE").write_text(f"{cfg['duration']}\n{time.strftime('%F %T')}\n")
    print(f"{name}: finished")


# --- evaluation ----------------------------------------------------------------------------------------------
def load_net(path):
    import pickle
    import torch
    edm2()
    with open(path, "rb") as f:
        net = pickle.load(f)["ema"]
    return net.to("cuda").eval().requires_grad_(False), torch


def evaluate(net, pr, batch=256):
    """D(x_t) [L, P, 3, 64, 64] float32 at the probes, in float32."""
    import torch
    L, P = pr["eps"].shape[:2]
    out = np.empty(pr["eps"].shape, np.float32)
    x0 = torch.from_numpy(pr["x0"]).cuda()
    with torch.no_grad():
        for l, s in enumerate(pr["sigmas"]):
            for lo in range(0, P, batch):
                e = torch.from_numpy(pr["eps"][l, lo:lo + batch]).cuda()
                x = x0[lo:lo + batch] + float(s) * e
                sig = torch.full((len(x),), float(s), device="cuda")
                out[l, lo:lo + batch] = net(x, sig, None, force_fp32=True).float().cpu().numpy()
    return out


def snapshots(run_dir):
    """{kimg: {ema: path}} of a run's complete snapshots past kimg 0. EDM2 pickles the snapshot before it saves
    the training state of the same step, so a training state at or past kimg (or DONE) proves it was written."""
    run_dir = Path(run_dir)
    states = [int(p.stem.split("-")[-1]) for p in run_dir.glob("training-state-*.pt")]
    upto = float("inf") if done_nimg(run_dir) else max(states, default=-1)
    out = {}
    for p in run_dir.glob("network-snapshot-*.pkl"):
        _, _, k, e = p.stem.split("-")
        if 0 < int(k) <= upto:
            out.setdefault(int(k), {})[e] = p
    return dict(sorted(out.items()))


def pred_path(out, name, kimg):
    return Path(out) / "preds" / name / f"k{kimg:07d}.npz"


def probe(a):
    """D(x_t) at the probes for every complete snapshot of every run, finished or not; rerun to pick up more."""
    cfg, pr = config(a.out), probes(a.out)
    emas = sorted(f"{e:.3f}" for e in cfg["ema"])
    for name, *_ in run_list(cfg):
        for kimg, paths in snapshots(Path(a.out) / "runs" / name).items():
            dst = pred_path(a.out, name, kimg)
            if dst.exists() or sorted(paths) != emas:
                continue
            atomic_save(dst, **{f"ema{e}": evaluate(load_net(paths[e])[0], pr) for e in emas})
            print(f"{name} @ {kimg} kimg", flush=True)


def ref(a):
    """tr Sigma = sigma^2 tr dD/dx_t of the reference network (Hutchinson, Rademacher probes), its standard
    error, the network's own squared error to x0 and its prediction."""
    import torch
    cfg, pr = config(a.out), probes(a.out)
    _edm = edm2()
    _edm.fetch(_edm.raw(cfg["ref"]))
    net, _ = _edm.edm_pickle(_edm.raw(cfg["ref"]))
    L, P = pr["eps"].shape[:2]
    R, B = cfg["hutchinson"], a.batch
    tr = np.empty((L, P, R))
    d_ref = np.empty(pr["eps"].shape, np.float32)
    x0 = torch.from_numpy(pr["x0"]).cuda()
    g = torch.Generator("cuda").manual_seed(SALT)
    for l, s in enumerate(pr["sigmas"]):
        for lo in range(0, P, B):
            x = (x0[lo:lo + B] + float(s) * torch.from_numpy(pr["eps"][l, lo:lo + B]).cuda()).requires_grad_(True)
            sig = torch.full((len(x),), float(s), device="cuda")
            d = net(x, sig, None, force_fp32=True).float()
            d_ref[l, lo:lo + B] = d.detach().cpu().numpy()
            for r in range(R):
                v = torch.randint(0, 2, x.shape, generator=g, device="cuda").float() * 2 - 1
                (jv,) = torch.autograd.grad((d * v).sum(), x, retain_graph=r < R - 1)
                tr[l, lo:lo + B, r] = (jv * v).flatten(1).sum(1).double().cpu().numpy()
        print(f"sigma {s}: tr Sigma median {np.median(s ** 2 * tr[l].mean(1)):.4g}", flush=True)
    s2 = pr["sigmas"][:, None] ** 2
    mmse = ((d_ref - pr["x0"][None]) ** 2).reshape(L, P, -1).sum(-1)
    atomic_save(Path(a.out) / "ref.npz", tr_sigma=s2 * tr.mean(-1), tr_se=s2 * tr.std(-1, ddof=1) / np.sqrt(R),
                mmse=mmse, d=d_ref, ref=np.asarray(cfg["ref"]))


def softmax_mean(xt, inv2s2, chunks):
    """sum_i w_i x_i / sum_i w_i with w_i = exp(-||xt - x_i||^2 / (2 sigma^2)), streamed over chunks of x [c, d]
    with a running max. Returns the mean [Q, d], sum w and sum w^2 [Q, 1] (both relative to the final max)."""
    import torch
    xt2 = (xt ** 2).sum(1, keepdim=True)
    m = torch.full((len(xt), 1), -torch.inf, device=xt.device)
    z, z2 = torch.zeros(len(xt), 1, device=xt.device), torch.zeros(len(xt), 1, device=xt.device)
    acc = torch.zeros_like(xt)
    for x in chunks:
        logit = -(xt2 - 2 * xt @ x.T + (x ** 2).sum(1)[None]) * inv2s2
        m_new = torch.maximum(m, logit.max(1, keepdim=True).values)
        a_old, w = torch.exp(m - m_new), torch.exp(logit - m_new)
        z = z * a_old + w.sum(1, keepdim=True)
        z2 = z2 * a_old ** 2 + (w ** 2).sum(1, keepdim=True)
        acc = acc * a_old + w @ x
        m = m_new
    return acc / z, z, z2


def exact(a):
    """Posterior mean of each subset's own N points at every probe, streamed over the subset with an online
    softmax, so no N x d matrix is ever held. Spread across `exact_subsets` subsets per probe."""
    import torch
    cfg, pr = config(a.out), probes(a.out)
    arr = np.load(train_array(), mmap_mode="r")
    L, P = pr["eps"].shape[:2]
    d = int(np.prod(pr["x0"].shape[1:]))
    xt = torch.from_numpy((pr["x0"][None] + pr["sigmas"][:, None, None, None, None] * pr["eps"])
                          .reshape(L * P, d).astype(np.float32)).cuda()          # sigmas are float64
    inv2s2 = torch.from_numpy(np.repeat(1 / (2 * pr["sigmas"] ** 2), P)).float().cuda()[:, None]
    for n in cfg["ns"]:
        dst = Path(a.out) / "exact" / f"n{n}.npz"
        if dst.exists():
            continue
        J, K = max(cfg["exact_subsets"], cfg["k"]), cfg["k"]
        s1 = torch.zeros(L * P, d, dtype=torch.float64, device="cuda")
        s2 = torch.zeros(L * P, dtype=torch.float64, device="cuda")
        ess = torch.zeros(L * P, dtype=torch.float64, device="cuda")
        keep = np.empty((K, L, P) + pr["x0"].shape[1:], np.float32)
        t0 = time.time()
        for j in range(J):
            idx = np.load(subset(a.out, n, j))
            chunks = (torch.from_numpy(np.ascontiguousarray(arr[idx[lo:lo + a.chunk]])).cuda()
                      .permute(0, 3, 1, 2).reshape(-1, d).float() / 127.5 - 1 for lo in range(0, n, a.chunk))
            mean, z, z2 = softmax_mean(xt, inv2s2, chunks)
            s1 += mean.double()
            s2 += (mean.double() ** 2).sum(1)
            ess += (z ** 2 / z2)[:, 0].double()
            if j < K:
                keep[j] = mean.reshape((L, P) + pr["x0"].shape[1:]).cpu().numpy()
        tr_cov = (s2 - (s1 ** 2).sum(1) / J) / (J - 1)
        atomic_save(dst, tr_cov=tr_cov.reshape(L, P).cpu().numpy(), ess=(ess / J).reshape(L, P).cpu().numpy(),
                    d=keep, subsets=np.asarray(J))
        print(f"n {n}: {J} subsets in {time.time() - t0:.0f}s", flush=True)


# --- analysis ------------------------------------------------------------------------------------------------
def gram(ds):
    """G [K, K, L, P] = <D_a, D_b> per probe, from ds [K, L, P, ...]."""
    f = ds.reshape(*ds.shape[:3], -1).astype(np.float64)
    return np.einsum("alpd,blpd->ablp", f, f)


def spread_from_gram(G, pick):
    """Unbiased tr Cov of the models `pick` (repeats allowed): sum_i ||D_i - mean||^2 / (k - 1)."""
    k = len(pick)
    sub = G[np.ix_(pick, pick)]
    return (np.einsum("iilp->lp", sub) - sub.sum((0, 1)) / k) / (k - 1)


def spread(ds):
    """Unbiased tr Cov over the first axis: ds [K, L, P, ...] -> [L, P]."""
    return spread_from_gram(gram(ds), np.arange(len(ds)))


def boot_median_ratio(ds, tr_sigma, B=2000, seed=0):
    """Median over probes of spread / tr Sigma, and its 5-95% interval over models resampled with replacement.
    A resample repeats models, which shrinks its spread by (k - 1) / k on average; replicates are scaled back.
    That average includes the resamples of one model repeated (spread 0), so they are kept, not skipped."""
    r = np.random.default_rng(seed)
    G, k = gram(ds), len(ds)
    point = np.median(spread_from_gram(G, np.arange(k)) / tr_sigma, -1)
    bs = np.asarray([np.median(spread_from_gram(G, r.integers(k, size=k)) * k / (k - 1) / tr_sigma, -1)
                     for _ in range(B)])
    return point, np.percentile(bs, 5, 0), np.percentile(bs, 95, 0)


def timing(run_dir):
    """Median sec/kimg, the last loss and the GPU, from an EDM2 run directory."""
    st = Path(run_dir) / "stats.jsonl"
    if not st.exists():
        return None
    rows = [json.loads(l) for l in st.read_text().splitlines() if l.strip()]
    spk = [r["Timing/sec_per_kimg"] for r in rows[1:] if "Timing/sec_per_kimg" in r]
    g = Path(run_dir) / "gpu.txt"
    gpu = sorted(set(g.read_text().split("\n")) - {""}) if g.exists() else []
    return dict(sec_per_kimg=float(np.median(spk)) if spk else np.nan, loss=rows[-1].get("Loss/loss", np.nan),
                kimg=rows[-1].get("Progress/kimg", np.nan), gpu=", ".join(gpu))


def analyze(a):
    cfg, pr = config(a.out), probes(a.out)
    out = Path(a.out)
    rf = np.load(out / "ref.npz")
    trs = rf["tr_sigma"]
    sig = pr["sigmas"]
    per = lambda v: ", ".join(f"{s:g}: {x:.3g}" for s, x in zip(sig, v))
    lines = [f"ImageNet-64 spread pilot ({out}), {time.strftime('%F %T')}",
             f"tr Sigma: {cfg['ref']} (Hutchinson, {cfg['hutchinson']} probes); median relative SE "
             + per(np.median(rf["tr_se"] / trs, -1)),
             "median over probes of tr Sigma: " + per(np.median(trs, -1)),
             "ref MSE / tr Sigma (1 = the reference is the posterior mean): "
             + per(np.median(rf["mmse"], -1) / np.median(trs, -1)), ""]
    res = dict(sigmas=sig)
    load = lambda nms, kimg, key: np.stack([np.load(pred_path(out, nm, kimg))[key] for nm in nms])
    ratio = lambda x: np.median((x ** 2).reshape(*x.shape[:3], -1).sum(-1) / trs, (0, 2))   # [K, L, P, ..] -> [L]
    for n in cfg["ns"]:
        names = [f"n{n}-j{j}-s{j}" for j in range(cfg["k"])]
        kimgs = sorted({int(p.stem[1:]) for nm in names for p in (out / "preds" / nm).glob("k*.npz")})
        ex = out / "exact" / f"n{n}.npz"
        z = np.load(ex) if ex.exists() else None
        for kimg in kimgs:
            have = [nm for nm in names if pred_path(out, nm, kimg).exists()]
            if len(have) < 2:
                continue
            for e in cfg["ema"]:
                key = f"ema{e:.3f}"
                ds = load(have, kimg, key)
                pt, lo, hi = boot_median_ratio(ds, trs)
                res[f"trained_n{n}_{key}_k{kimg}"] = np.stack([pt, lo, hi])
                lines.append(f"N={n:>7,} {kimg * 1000 / 2 ** 20:4.3g} Mi EMA {e:.3f}, {len(have)} models: "
                             "spread / tr Sigma (median over probes, 5-95% over models) "
                             + ", ".join(f"{s:g}: {p:.3g} [{l_:.3g}, {h:.3g}]" for s, p, l_, h in zip(sig, pt, lo, hi)))
                lines.append(f"{'':>9} model MSE to x0 / tr Sigma: " + per(ratio(ds - pr["x0"][None, None])))
                if z is not None:
                    own = z["d"][[int(nm.split("-j")[1].split("-")[0]) for nm in have]]
                    lines.append(f"{'':>9} model vs its own subset's exact mean, ||D - D_N||^2 / tr Sigma: "
                                 + per(ratio(ds - own)))
                for n2, j, seeds in cfg["seed_only"]:
                    nms = [f"n{n}-j{j}-s{s}" for s in [j] + list(seeds)]
                    if n2 == n and all(pred_path(out, nm, kimg).exists() for nm in nms):
                        so = boot_median_ratio(load(nms, kimg, key), trs)
                        res[f"seedonly_n{n}_{key}_k{kimg}"] = np.stack(so)
                        lines.append(f"{'':>9} seed only (subset {j}, {len(nms)} seeds): " + per(so[0]))
        if z is not None:
            res[f"exact_n{n}"] = np.median(z["tr_cov"] / trs, -1)
            lines.append(f"N={n:>7,} exact mean of N points ({int(z['subsets'])} subsets): " + per(res[f"exact_n{n}"]))
            lines.append(f"{'':>9} effective number of points in the exact mean (median): "
                         + per(np.median(z["ess"], -1)))
        lines.append("")
    lines.append("training (median sec/kimg, last loss, kimg):")
    for name, *_ in run_list(cfg):
        t = timing(out / "runs" / name)
        if t:
            lines.append(f"  {name:>18}: {t['sec_per_kimg']:.3f} s/kimg -> "
                         f"{t['sec_per_kimg'] * cfg['duration'] / 1000 / 3600:.2f} h per run, loss {t['loss']:.4f}, "
                         f"{t['kimg']:.0f} kimg, done {done_nimg(out / 'runs' / name) >= cfg['duration']}, {t['gpu']}")
    np.savez(out / "analysis.npz", **res)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figure(cfg, res, out / "fig")


def figure(cfg, res, fig_dir):
    """spread-in64: spread / tr Sigma against N at the longest training every N has reached, one panel per sigma.
    spread-in64-length: the same against training length, one line per N (has it converged?)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(exist_ok=True)
    sig, ns = res["sigmas"], np.asarray(cfg["ns"])
    e = f"ema{cfg['ema'][0]:.3f}"
    got = lambda n, k: res.get(f"trained_n{n}_{e}_k{k}")
    ks = sorted({int(key.rsplit("_k", 1)[1]) for key in res if key.startswith("trained_")})
    if not ks:
        print("no trained spreads yet")
        return
    common = [k for k in ks if all(got(n, k) is not None for n in ns)]
    k = (common or ks)[-1]
    nan = np.full((3, len(sig)), np.nan)
    mi = lambda k_: k_ * 1000 / 2 ** 20

    fig, axs = plt.subplots(1, len(sig), figsize=(2.1 * len(sig), 2.4), sharey=True, constrained_layout=True,
                            squeeze=False)
    for l, (ax, s) in enumerate(zip(axs[0], sig)):
        tr = np.array([(got(n, k) if got(n, k) is not None else nan)[:, l] for n in ns])
        ax.errorbar(ns, tr[:, 0], yerr=[tr[:, 0] - tr[:, 1], tr[:, 2] - tr[:, 0]], color="#7b4fb0", marker="o",
                    ms=4, lw=1.5, capsize=2, label="trained (Adam)")
        ex = np.array([res.get(f"exact_n{n}", np.full(len(sig), np.nan))[l] for n in ns])
        ax.plot(ns, ex, color="0.6", lw=4, alpha=0.6, label="exact mean of N points")
        for n2, _, _ in cfg["seed_only"]:
            so = res.get(f"seedonly_n{n2}_{e}_k{k}")
            if so is not None:
                ax.plot([n2], [so[0, l]], "D", mfc="white", mec="#7b4fb0", ms=5, label="seed only")
        if np.isfinite(tr[0, 0]):
            ax.plot(ns, tr[0, 0] * ns[0] / ns, "k--", lw=0.8, label="1/N")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"$\\sigma={s:g}$", fontsize=9)
        ax.set_xlabel("N")
    axs[0, 0].set_ylabel(r"spread / tr $\Sigma$")
    axs[0, -1].legend(fontsize=6, frameon=False)
    fig.suptitle(f"ImageNet-64, {mi(k):.3g} Mi training images, EMA {e[3:]}", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"spread-in64.{ext}", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(sig), figsize=(2.1 * len(sig), 2.4), sharey=True, constrained_layout=True,
                            squeeze=False)
    for l, (ax, s) in enumerate(zip(axs[0], sig)):
        for i, n in enumerate(ns):
            kk = [k_ for k_ in ks if got(n, k_) is not None]
            if kk:
                ax.plot([mi(k_) for k_ in kk], [got(n, k_)[0, l] for k_ in kk], marker="o", ms=3,
                        color=plt.cm.viridis(i / max(len(ns) - 1, 1)), label=f"N={n:,}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"$\\sigma={s:g}$", fontsize=9)
        ax.set_xlabel("Mi training images")
    axs[0, 0].set_ylabel(r"spread / tr $\Sigma$")
    axs[0, -1].legend(fontsize=6, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"spread-in64-length.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {fig_dir / 'spread-in64.pdf'} ({mi(k):.3g} Mi) and spread-in64-length.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["prep", "train", "probe", "ref", "exact", "analyze", "runs"])
    ap.add_argument("--out")
    ap.add_argument("--src", default=os.environ.get("VMG_IMAGENET_TRAIN"),
                    help="prep: ImageNet train (ILSVRC2012_img_train), one folder per synset; default $VMG_IMAGENET_TRAIN")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--per-class", type=int, default=0,
                    help="prep: only the first K images of each class, into img64_train_u8-kK.npy (smoke)")
    ap.add_argument("--task", type=int)
    ap.add_argument("--batch", type=int, default=64, help="ref: probes per backward")
    ap.add_argument("--chunk", type=int, default=8192, help="exact: train images per matmul")
    a = ap.parse_args()
    if a.stage == "prep":
        return prep(a)
    if a.stage == "runs":
        for i, r in enumerate(run_list(config(a.out))):
            print(i, *r)
        return
    {"train": train, "probe": probe, "ref": ref, "exact": exact, "analyze": analyze}[a.stage](a)


if __name__ == "__main__":
    main()
