#!/usr/bin/env python
"""Guidance between two training runs of the same model on ImageNet-64 (EDM2-XS).

The paper says guidance should work between instances of one model that differ only in their training
randomness (init, data order, noise draws). Three conditional EDM2-XS instances at 32 Mi training images:
A and B are ours (stock train_edm2.py --preset=edm2-img64-xs, seeds below), N is NVIDIA's released run
(raw snapshot edm2-img64-xs-0033554); Nu is NVIDIA's unconditional XS at the same length (CFG).

    python reproduce/experiments/spread_in64.py prep --src <ILSVRC2012 train>   # CPU, once: the train array
    python reproduce/experiments/seedpair_in64.py prep         # CPU: train array (or LMDB) -> EDM2 dataset zip
    python reproduce/experiments/seedpair_in64.py refcompare   # CPU: our zip's reference stats vs NVIDIA's img64.pkl
    python reproduce/experiments/seedpair_in64.py labelcheck   # GPU: LMDB labels follow NVIDIA's class indices
    python reproduce/experiments/seedpair_in64.py gate --ngpu 4  # GPU: labelcheck + ref stats; train only if it passes
    sbatch ... reproduce/slurm/train_seedpair.sbatch A           # GPU: the gate, then stock train_edm2.py
    python reproduce/experiments/seedpair_in64.py pickles      # CPU: our snapshots' config vs NVIDIA's
    python reproduce/experiments/seedpair_in64.py mse          # GPU: per-sigma denoising MSE on val, every instance
    python reproduce/experiments/ens_edm2.py moments --out $VMG_RUNS/edm2-64-seedpair --tag seedpair --n 10000 \
        --models A,B,N,Nu --ckpt A=<pkl> ...                    # GPU: Gram of every member and x0
    python reproduce/experiments/seedpair_in64.py fit          # CPU: VM w of every pair (vmg.solve), error corr
    python reproduce/experiments/ens_edm2.py sample/eval --out ... --ckpt ...   # GPU: arms, FID / FD_DINOv2
    python reproduce/experiments/seedpair_in64.py analyze      # CPU: summary.txt
    python reproduce/experiments/seedpair_in64.py arms --list L   # the arms of fig:cross-edm2xs, one line each, for
        sbatch ... reproduce/slurm/seedpair_worker.sbatch L      # GPU workers that sample and score them
    python reproduce/experiments/seedpair_in64.py table        # CPU: metrics/sw-*.json -> tables/arms.csv (the figure's)

The dataset is written the way EDM2's dataset_tool.py convert --transform=center-crop-dhariwal
--resolution=64x64 writes it (same transform, PNG settings, archive names and dataset.json); only the
source differs: the train set is read from spread_in64.py prep's array ($VMG_IMAGENET/img64_train_u8.npy, the
same crop of the official train JPEGs in sorted synset listing order), or, when $VMG_IMAGENET_TRAIN_LMDB is set,
from an LMDB of pickled {"image": JPEG bytes, "label": int} under keys str(i), with the count under
"num_samples" (the paper's runs; its record order fixes which images the zip rows hold). The gate checks either
against NVIDIA's reference statistics.
"""
import argparse
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LMDB = os.environ.get("VMG_IMAGENET_TRAIN_LMDB", "")   # the ImageNet train set as an LMDB (see prep)
M_TRAIN = 1281167
SEEDS = {"A": 20261117, "B": 31415926}         # far apart: EDM2 seeds rank r with seed * world + r
TRAIN = {**SEEDS, "S1": 27182818}               # every run we train; S1 is an EDM2-S (train_seedpair.sbatch picks the preset)
KIMG = 33554                                    # 32 Mi, NVIDIA's raw snapshot edm2-img64-xs-0033554


def out_root():
    from vmg import runs
    d = runs.root() / "edm2-64-seedpair"
    d.mkdir(parents=True, exist_ok=True)
    return d


def zip_path():
    return out_root() / "img64" / "img64.zip"


def edm2_on_path():
    from vmg.models import _edm
    _edm.edm_setup()
    return _edm


# --- data ------------------------------------------------------------------------------------------------------
_T = None
_ENV = None


def _lmdb():
    if os.environ.get("VMG_PYLIB_LMDB"):                   # where the lmdb package is, if not in the env
        sys.path.append(os.environ["VMG_PYLIB_LMDB"])
    import lmdb
    assert LMDB, "set VMG_IMAGENET_TRAIN_LMDB to the ImageNet train LMDB"
    # readahead must be on: with it off every 4 KB page of the 150 GB file is a separate read
    return lmdb.open(LMDB, readonly=True, lock=False, readahead=True, meminit=False)


def _png(img):
    """uint8 [64, 64, 3] -> PNG bytes (compress_level 0, as dataset_tool's convert())."""
    import PIL.Image
    bits = io.BytesIO()
    PIL.Image.fromarray(np.asarray(img), "RGB").save(bits, format="png", compress_level=0, optimize=False)
    return bytes(bits.getbuffer())


def _encode(jpeg):
    """JPEG bytes -> PNG bytes of dataset_tool's center-crop-dhariwal 64x64."""
    global _T
    import PIL.Image
    if _T is None:
        edm2_on_path()
        import dataset_tool
        _T = dataset_tool.make_transform("center-crop-dhariwal", 64, 64)
    return _png(_T(np.array(PIL.Image.open(io.BytesIO(jpeg)).convert("RGB"))))


_ARR = None
PER_CLASS = 0          # prep --per-class: the smoke-sized train array (set before the worker pool forks)


def _read_array(span):
    """[(i, label, png)] for rows lo..hi-1 of spread_in64.py prep's train array (already cropped by dataset_tool)."""
    global _ARR
    if _ARR is None:
        from reproduce.experiments import spread_in64
        _ARR = (np.load(spread_in64.train_array(PER_CLASS), mmap_mode="r"),
                np.load(spread_in64.train_labels(PER_CLASS)))
    X, y = _ARR
    return [(i, int(y[i]), _png(X[i])) for i in range(*span)]


def _read_encode(span):
    """[(i, label, png)] for records lo..hi-1, read by this worker itself: one read stream per worker, since a
    single reader is the bottleneck."""
    global _ENV
    if _ENV is None:
        _ENV = _lmdb()
    with _ENV.begin() as txn:
        recs = [(i, pickle.loads(txn.get(str(i).encode()))) for i in range(*span)]
    return [(i, int(r["label"]), _encode(r["image"])) for i, r in recs]


def prep(a):
    global PER_CLASS
    from multiprocessing import Pool
    edm2_on_path()
    import dataset_tool
    dest = zip_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(".img64.tmp.zip")
    if LMDB:
        env = _lmdb()
        with env.begin() as txn:                       # keys are str(i), unpadded, plus "num_samples"
            raw = txn.get(b"num_samples")
        env.close()                                    # workers open their own (lmdb refuses an inherited one)
        try:
            n = int(pickle.loads(raw))
        except Exception:
            n = int(raw.decode())
        read = _read_encode
    else:
        from reproduce.experiments import spread_in64
        PER_CLASS = a.per_class
        arr = spread_in64.train_array(PER_CLASS)
        assert arr.exists(), (f"{arr} missing: run reproduce/experiments/spread_in64.py prep --src <ILSVRC2012 train>"
                              + (f" --per-class {PER_CLASS}" if PER_CLASS else "") + " (or set VMG_IMAGENET_TRAIN_LMDB)")
        n, read = len(np.load(arr, mmap_mode="r")), _read_array
    assert n == (1000 * a.per_class if a.per_class else M_TRAIN), n
    root_dir, save_bytes, close_dest = dataset_tool.open_dest(str(tmp))
    labels, t0 = [], time.time()
    spans = [(lo, min(n, lo + a.chunk)) for lo in range(0, n, a.chunk)]
    with Pool(a.workers) as pool:
        for k, part in enumerate(pool.imap(read, spans)):    # in order: the zip is written sequentially
            for i, lab, png in part:
                s = f"{i:08d}"
                fname = f"{s[:5]}/img{s}.png"
                save_bytes(os.path.join(root_dir, fname), png)
                labels.append([fname, lab])
            if k % 20 == 0:
                print(f"{len(labels):,} / {n:,}  {time.time() - t0:.0f}s", flush=True)
    assert len(labels) == n
    lab = np.array([l for _, l in labels])
    assert lab.min() == 0 and lab.max() == 999 and len(np.unique(lab)) == 1000, (lab.min(), lab.max())
    save_bytes(os.path.join(root_dir, "dataset.json"), json.dumps({"labels": labels}))
    close_dest()
    os.replace(tmp, dest)
    np.save(dest.with_name("labels.npy"), lab.astype(np.int16))
    print(f"wrote {dest} ({dest.stat().st_size / 2**30:.1f} GiB); per class {np.bincount(lab).min()}-"
          f"{np.bincount(lab).max()}")


def refcompare(a):
    """Our zip's EDM2 reference statistics (calculate_metrics.py ref) against NVIDIA's dataset-refs/img64.pkl.
    Returns {detector: FD(ours, NVIDIA)}."""
    from vmg.models import _edm
    edm2_on_path()
    ours = pickle.load(open(a.ref, "rb"))
    out = {}
    theirs = pickle.load(open(_edm.fetch("edm2/dataset-refs/img64.pkl"), "rb"))
    import scipy.linalg
    for det in sorted(k for k, v in theirs.items() if isinstance(v, dict)):      # also holds a plain count
        mu0, s0 = (np.asarray(theirs[det][k], np.float64) for k in ("mu", "sigma"))
        mu1, s1 = (np.asarray(ours[det][k], np.float64) for k in ("mu", "sigma"))
        fd = float(np.real(np.square(mu1 - mu0).sum() + np.trace(s1 + s0 - 2 * scipy.linalg.sqrtm(s1 @ s0))))
        print(f"{det}: |dmu|/|mu| {np.linalg.norm(mu1 - mu0) / np.linalg.norm(mu0):.2e}, "
              f"|dSigma|/|Sigma| {np.linalg.norm(s1 - s0) / np.linalg.norm(s0):.2e}, FD(ours, NVIDIA) {fd:.5f}")
        out[det] = fd
    return out


# --- checks against NVIDIA's run --------------------------------------------------------------------------------
def zip_images(rows):
    """uint8 [n, 3, 64, 64] and labels of the given zip rows."""
    import zipfile
    import PIL.Image
    z = zipfile.ZipFile(zip_path())
    lab = np.load(zip_path().with_name("labels.npy"))
    ims = [np.asarray(PIL.Image.open(io.BytesIO(z.read(f"{r:08d}"[:5] + f"/img{r:08d}.png")))) for r in rows]
    return np.stack(ims).transpose(0, 3, 1, 2), lab[rows].astype(np.int64)


def labelcheck(a):
    """With NVIDIA's XS (2 Gi), the denoising error at the LMDB label must be clearly below the error at a
    random label: a permuted class mapping would make the two equal."""
    import torch
    _edm = edm2_on_path()
    _edm.fetch(_edm.raw("edm2-img64-xs-2147483-0.100"))
    net, _ = _edm.edm_pickle(_edm.raw("edm2-img64-xs-2147483-0.100"))
    rows = np.random.default_rng(0).choice(M_TRAIN, a.n, replace=False)
    u8, lab = zip_images(np.sort(rows))
    x0 = torch.from_numpy(u8).cuda().float() / 127.5 - 1
    g = torch.Generator(device="cuda").manual_seed(0)
    eye = torch.eye(1000, device="cuda")
    frac = None
    wrong = torch.from_numpy((lab + np.random.default_rng(1).integers(1, 1000, len(lab))) % 1000).cuda()
    for s in (0.5, 1.0, 2.0):
        e1, e0 = [], []
        for lo in range(0, len(x0), 250):
            x = x0[lo:lo + 250]
            xt = x + s * torch.randn(x.shape, generator=g, device="cuda")
            t = torch.full((len(x),), s, device="cuda")
            with torch.no_grad():
                for L, acc in ((eye[torch.from_numpy(lab[lo:lo + 250]).cuda()], e1), (eye[wrong[lo:lo + 250]], e0)):
                    acc.append(((net(xt, t, L, force_fp32=True) - x) ** 2).flatten(1).mean(1).cpu())
        e1, e0 = torch.cat(e1), torch.cat(e0)
        print(f"sigma {s}: MSE true label {e1.mean():.5f}, random label {e0.mean():.5f}, "
              f"true better on {(e1 < e0).float().mean():.1%} of {len(e1)} images", flush=True)
        if s == 1.0:
            frac = float((e1 < e0).float().mean())
    return frac


def gate(a):
    """Before any training: the labels follow NVIDIA's class indices and our zip has NVIDIA's reference statistics
    (so crop, images and set are identical). Writes gate.json; exits 1 if either fails. Idempotent."""
    import subprocess
    g = out_root() / "gate.json"
    if g.exists() and json.loads(g.read_text())["pass"]:
        print("gate passed earlier:", g.read_text())
        return
    frac = labelcheck(a)
    ref = out_root() / "img64-ours-ref.pkl"
    if not ref.exists():
        _edm = edm2_on_path()
        tmp = ref.with_name(f".ref.tmp{os.getpid()}.pkl")
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        f"--nproc_per_node={a.ngpu}", str(_edm.src("edm2") / "calculate_metrics.py"), "ref",
                        f"--data={zip_path()}", f"--dest={tmp}", "--batch=256", "--workers=4"], check=True,
                       cwd=str(_edm.src("edm2")))
        os.replace(tmp, ref)
    a.ref = str(ref)
    fd = refcompare(a)
    ok = frac > 0.9 and fd["fid"] < 0.1 and fd["fd_dinov2"] < 2.0
    g.write_text(json.dumps(dict(label_true_better=frac, fd=fd, **{"pass": ok}, time=time.strftime("%F %T"))))
    print("gate", "PASSED" if ok else "FAILED", g.read_text(), flush=True)
    if not ok:
        sys.exit(1)


def snapshot(name, ema):
    """Checkpoint path of instance `name` at 32 Mi with EMA `ema` ('0.050' / '0.100')."""
    from vmg.models import _edm
    nv = {"N": "edm2-img64-xs", "Nu": "edm2-img64-xs-uncond", "Sn": "edm2-img64-s"}     # NVIDIA's: fetched on use
    if name in nv:
        return _edm.fetch(_edm.raw(f"{nv[name]}-{KIMG:07d}-{ema}"))
    return out_root() / "train" / name / f"network-snapshot-{KIMG:07d}-{ema}.pkl"


def _describe(path):
    edm2_on_path()
    d = pickle.load(open(path, "rb"))
    net = d["ema"]

    def init_kwargs(m):
        return {k: v for k, v in dict(getattr(m, "init_kwargs", {})).items()}

    loss = {k: v for k, v in vars(d["loss_fn"]).items() if not k.startswith("_")} if "loss_fn" in d else None
    return dict(keys=sorted(d), net_class=type(net).__name__, net_kwargs=init_kwargs(net),
                unet_kwargs=init_kwargs(net.unet) if hasattr(net, "unet") else None,
                params=int(sum(p.numel() for p in net.parameters())), loss=loss,
                dataset={k: v for k, v in dict(d.get("dataset_kwargs") or {}).items()},
                encoder=(type(d["encoder"]).__name__, init_kwargs(d["encoder"])) if d.get("encoder") else None)


def pickles(a):
    """Every field our 32 Mi snapshots pickle (config, loss, dataset, encoder, network kwargs, parameter count)
    against NVIDIA's at the same length. Only the dataset path may differ."""
    refs = {}
    for name in TRAIN:
        nv = "Sn" if name.startswith("S") else "N"            # each run against NVIDIA's model of the same size
        if nv not in refs:
            refs[nv] = _describe(snapshot(nv, a.ema))
            print(f"NVIDIA {nv}:", json.dumps(refs[nv], indent=1, default=str))
        ref = refs[nv]
        p = snapshot(name, a.ema)
        if not p.exists():
            print(f"{name}: {p} missing")
            continue
        d = _describe(p)
        diff = {k: (d[k], ref[k]) for k in d if d[k] != ref[k]}
        print(f"{name}: differences from NVIDIA:", json.dumps(diff, indent=1, default=str) if diff else "none")
        opts = json.loads((p.parent / "training_options.json").read_text())
        print(f"{name}: seed {opts['seed']}, batch {opts['batch_size']}, lr {opts['lr_kwargs']}, "
              f"loss {opts['loss_kwargs']}, net {opts['network_kwargs']}")


def mse(a):
    """Denoising MSE of each instance on 10k ImageNet val images at the 32 sampler levels (fp32)."""
    import torch
    from vmg.data import imagenet
    _edm = edm2_on_path()
    from vmg.models.edm2 import EDMHeun
    sig = EDMHeun().levels()["sigma"]
    rows = imagenet.calib_rows(a.n)
    x0all = imagenet.pixels(rows)
    lab = torch.nn.functional.one_hot(imagenet.class_of(rows, 64), 1000).float()
    res = {}
    names = [n for n in a.names.split(",") if snapshot(n, a.ema).exists()]
    for name in names:
        net = pickle.load(open(snapshot(name, a.ema), "rb"))["ema"].cuda().eval().requires_grad_(False)
        m = np.zeros(len(sig))
        for k, s in enumerate(sig):
            g = torch.Generator().manual_seed(1000 + k)
            for lo in range(0, len(rows), 500):
                x = x0all[lo:lo + 500].cuda()
                xt = x + float(s) * torch.randn(x.shape, generator=g).cuda()
                t = torch.full((len(x),), float(s), device="cuda")
                with torch.no_grad():
                    d = net(xt, t, lab[lo:lo + 500].cuda() if net.label_dim else None, force_fp32=True)
                m[k] += float(((d - x) ** 2).sum())
        res[name] = m / x0all.numel()
        print(name, " ".join(f"{v:.4g}" for v in res[name]), flush=True)
    np.savez(out_root() / f"mse-ema{a.ema}.npz", sigma=sig, **res)
    for name in names:
        if name != "N" and "N" in res:
            r = res[name] / res["N"]
            print(f"{name}/N: median {np.median(r):.4f}, range {r.min():.4f}-{r.max():.4f}")


# --- variance-matched weights from ens_edm2.py's moments pass ---------------------------------------------------
PAIRS = [("A", "N"), ("N", "A"), ("B", "N"), ("N", "B"), ("A", "B"), ("B", "A"),     # seed pairs (D1 <- D0)
         ("N", "Nu"), ("A", "Nu"), ("B", "Nu")]                                       # CFG with NVIDIA's uncond


def load_gram(tag):
    """Sum the shards of ens_edm2.py moments --tag TAG: names, sigma, gram [S, K+1, K+1], mean [S, K+1, d], n, d
    (gram/mean per sample, Y = [D_1 .. D_K, x0])."""
    paths = sorted(out_root().glob(f"moments-{tag}-*of*.npz"))
    assert paths, f"no moments-{tag}-* in {out_root()}"
    zs = [np.load(p) for p in paths]
    assert len(zs) == int(paths[0].stem.split("of")[-1]), f"{len(zs)} shards of {paths[0].stem}"
    z0 = zs[0]
    return dict(names=list(z0["names"]), sigma=z0["sigma"], n=int(z0["n"]), d=int(z0["d"]),
                gram=sum(z["gram"] for z in zs), mean=sum(z["mean"] for z in zs))


def pair_moments(g, d1, d0):
    """solve.Moments of the pair (D1 = d1, D0 = d0) from the Gram of every member and x0."""
    from vmg import solve
    i0, i1, x = g["names"].index(d0), g["names"].index(d1), len(g["names"])
    G, m, d = g["gram"], g["mean"], g["d"]
    r = lambda i, j: G[:, i, j] - G[:, i, x] - G[:, j, x] + G[:, x, x]                 # E<r_i, r_j>
    rb = {i: m[:, i] - m[:, x] for i in (i0, i1)}                                       # E r_i
    dot = lambda u, v: np.einsum("sd,sd->s", u, v)
    rr = np.stack([r(i0, i0), r(i0, i1), r(i1, i1)], 1)
    mean_rr = np.stack([dot(rb[i0], rb[i0]), dot(rb[i0], rb[i1]), dot(rb[i1], rb[i1])], 1)
    xr = np.stack([G[:, x, i] - G[:, x, x] - dot(m[:, x], rb[i]) for i in (i0, i1)], 1)  # E<x0 - xbar, r_i>
    var_x0 = float(np.mean(G[:, x, x] - dot(m[:, x], m[:, x])))
    return solve.Moments(sigma=g["sigma"], t=g["sigma"], var_x0=var_x0 / d, xr=xr / d, rr=rr / d,
                         rr_centred=(rr - mean_rr) / d, n=g["n"], d=d)


def fit(a):
    """Global and per-level variance-matched w (vmg.solve, the paper's default rule) of every pair present, the
    MSE-optimal w, and the error correlation corr(r_1, r_0) per level. Writes w-<tag>.json."""
    from vmg import solve
    g = load_gram(a.tag)
    res, lines = {}, []
    for d1, d0 in PAIRS:
        if d1 not in g["names"] or d0 not in g["names"]:
            continue
        m = pair_moments(g, d1, d0)
        sch = solve.fit_schedule(m)
        wg, ok = solve.fit_global(m)
        c00, c01, c11 = m.rr_centred.T
        corr = c01 / np.sqrt(c00 * c11)
        res[f"{d1}<-{d0}"] = dict(w_global=wg, admissible=ok, w_schedule=sch["w"].tolist(),
                                  schedule_admissible=sch["admissible"].tolist(),
                                  w_mse=solve.w_mse(m).tolist(), err_corr=corr.tolist(),
                                  mse1=(m.rr[:, 2]).tolist(), mse0=(m.rr[:, 0]).tolist())
        lines.append(f"{d1:>2} <- {d0:<2}: global w {wg:.3f}{'' if ok else ' (clipped)'}; schedule "
                     f"{np.min(sch['w']):.2f}-{np.max(sch['w']):.2f}; err corr median {np.median(corr):.3f} "
                     f"(range {corr.min():.3f}-{corr.max():.3f}); MSE1/MSE0 median {np.median(m.rr[:, 2] / m.rr[:, 0]):.3f}")
    res["sigma"] = g["sigma"].tolist()
    (out_root() / f"w-{a.tag}.json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


def analyze(a):
    """Every metrics/<arm>.json as a table: FID, FD_DINOv2, precision and recall in both spaces (Inception,
    DINOv2), and the arm's members and w. Arms are grouped by name prefix and sorted by w within a group."""
    from vmg.metrics.image import DETECTORS
    rows = []
    for p in sorted((out_root() / "metrics").glob("*.json")):
        r = json.loads(p.read_text())
        ar = r["args"]
        try:
            w = float(ar["w"])
        except ValueError:
            w = np.nan
        pr = [r.get(f"{m}_{d}", np.nan) for d in DETECTORS for m in ("precision", "recall")]
        rows.append((r["arm"].split("-w")[0], w, r["arm"], r["n"], r.get("fid", np.nan), r.get("fd_dinov2", np.nan),
                     pr, f"{ar['d1']}" + (f" <- {ar['d0']} w {ar['w'][:40]}" if ar.get("d0") else "")))
    rows.sort(key=lambda t: (t[0], t[1], t[3]))
    pr_h = " ".join(f"{m[0].upper()}_{d[:4]:<4}" for d in DETECTORS for m in ("precision", "recall"))
    lines = [f"{'arm':<26} {'n':>6} {'FID':>7} {'FD_DINOv2':>10} {pr_h}  what"]
    for g, w, arm, n, fid, fd, pr, what in rows:
        lines.append(f"{arm:<26} {n:>6} {fid:7.3f} {fd:10.2f} " + " ".join(f"{v:6.3f}" for v in pr) + f"  {what}")
    (out_root() / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# fig:cross-edm2xs: D = D_B + w (D_A - D_B) at three EMA lengths (post-hoc 0.035, raw 0.050 and 0.100)
FIG_ARMS = [("p035", "p0.035", [x / 4 for x in range(-4, 9)])] + \
    [(f"e{e[2:]}", e, [x / 4 for x in range(-8, 9)]) for e in ("0.050", "0.100")]
ARM_SOURCE = {"000": "raw training weights (no EMA)", "050": "EMA saved during training",
              "100": "EMA saved during training"}
METRIC_KEYS = ["fid", "fd_dinov2", "precision_inception", "recall_inception", "density_inception", "coverage_inception",
               "precision_dinov2", "recall_dinov2", "density_dinov2", "coverage_dinov2"]


def arms(a):
    """The worker list of fig:cross-edm2xs (reproduce/slurm/seedpair_worker.sbatch reads it): "ARM EMA ARGS"."""
    lines = [f"sw-AB-{tag}-w{w:g} {ema} --d1 A --d0 B --w={w:g}" for tag, ema, ws in FIG_ARMS for w in ws]
    lines = lines[:1] if a.smoke else lines
    Path(a.list).parent.mkdir(parents=True, exist_ok=True)
    Path(a.list).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.list}: {len(lines)} arms")


def table(a):
    """metrics/sw-*.json (ens_edm2.py eval) -> tables/arms.csv, one row per arm, as fig_cross_seedpair.py reads it."""
    import csv
    import glob
    import json
    import re
    rows = []
    for p in glob.glob(str(out_root() / "metrics" / "sw-*.json")):
        r = json.load(open(p))
        m = re.match(r"sw-(A[NB])-[ep](\d+)-w(.*)", r["arm"])
        if not m:
            continue
        rows.append(dict(arm=r["arm"], d1="A", guide=m[1][1], ema=f"0.{m[2]}" if m[2] != "000" else "0",
                         ema_source=ARM_SOURCE.get(m[2], "post-hoc reconstruction"), w=float(m[3]), n=r["n"],
                         **{k: r[k] for k in METRIC_KEYS}))
    assert rows, f"no sw-*.json under {out_root() / 'metrics'}"
    rows.sort(key=lambda t: (t["guide"], float(t["ema"]), t["w"]))
    path = out_root() / "tables" / "arms.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} arms")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["prep", "refcompare", "labelcheck", "gate", "pickles", "mse", "fit", "analyze",
                                      "arms", "table"])
    ap.add_argument("--list", help="arms: where to write the worker list")
    ap.add_argument("--per-class", type=int, default=0, help="prep: the zip of spread_in64.py prep --per-class K (smoke)")
    ap.add_argument("--smoke", action="store_true", help="arms: the first arm only")
    ap.add_argument("--tag", default="seedpair", help="fit: the ens_edm2.py moments --tag")
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--ref", help="refcompare: our calculate_metrics.py ref pickle")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--ema", default="0.100")
    ap.add_argument("--names", default="A,B,N")
    a = ap.parse_args()
    {"prep": prep, "refcompare": refcompare, "labelcheck": labelcheck, "gate": gate, "pickles": pickles,
     "mse": mse, "fit": fit, "analyze": analyze, "arms": arms, "table": table}[a.stage](a)


if __name__ == "__main__":
    main()
