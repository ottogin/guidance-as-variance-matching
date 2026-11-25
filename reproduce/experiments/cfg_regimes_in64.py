#!/usr/bin/env python
"""CFG on ImageNet-64: label information against model error, and where guidance helps (fig:cfg-regimes, App. B.3).

    python reproduce/experiments/seedpair_in64.py prep                           # CPU: the train set as img64.zip
    sbatch ... reproduce/slurm/train_xs_2e27.sbatch DA 0 0.1                     # GPU: also DB 1 0.1, A 0 and B 1
    python reproduce/experiments/cfg_regimes_in64.py gram --shard K --shards 4   # GPU: the Gram, levels K mod 4
    python reproduce/experiments/cfg_regimes_in64.py sweep --point P             # GPU: one FID point, P = 0..11
    python reproduce/experiments/cfg_regimes_in64.py analyze                     # CPU: summary.json, App. B.3's numbers
    python reproduce/experiments/cfg_regimes_in64.py test                        # CPU: the code paths on tiny networks
    python reproduce/figures/fig_cfg_regimes.py --out DIR/cfg-regimes-img64      # CPU: the figure

Models: EDM2-XS (edm2-img64-xs, batch 2,048, 2^27 training images, the EMA of length 0.100 that training keeps)
trained with classifier-free label dropout p = 0.1, seeds 0 and 1 (DA, DB). A zero label vector is EDM2's
unconditional input, so each network gives both branches of CFG: D(x_t, c) with the one-hot label and D(x_t) with
zeros. A and B are the same preset and seeds without dropout; they enter one number of App. B.3, the seed-to-seed
distance of the conditional predictions. NVIDIA's EDM2-XL (EMA 0.040) is the proxy for the posterior mean, and
EDM2-L (0.040) bounds the error of XL: where L has the lower denoising MSE, E|D_XL - D*|^2 >= MSE_XL - MSE_L.

gram: 4,096 training images with their labels (rows drawn with seed 123 from img64.zip), each noised once at each
of the 32 levels of EDM2's sampler (noise seed 1000 + level, the same for every branch). Per level, with xbar the
mean of the images,
    G[k, l] = sum <D_k - xbar, D_l - xbar>,   H[k] = sum <D_k - xbar, x0 - xbar>,   X2 = sum |x0 - xbar|^2,
so every squared distance, inner product and denoising MSE of the branches follows. The networks run at the
release's precision (fp16 inside the UNet), as the paper's pass did.

sweep: EDM2's deterministic 32-step Heun sampler (vmg.models.edm2.EDMHeun, bitwise edm_sampler) on seeds 0..9,999,
whose noise and class come from generate_images.py's per-seed generator. Guidance D(x_t) + w (D(x_t, c) - D(x_t))
with w = 2 acts only at the denoiser calls whose sigma lies in one window (WINDOWS), or nowhere ("none"). FID and
FD_DINOv2 against NVIDIA's ImageNet-64 training statistics (edm2/dataset-refs/img64.pkl).

analyze: per level, for each seed, |Delta|^2 with Delta = D(x_t, c) - D(x_t), the conditional error against XL, the
label's MSE gain E|D(x_t) - x0|^2 - E|D(x_t, c) - x0|^2; across the seeds, the systematic part <Delta_A, Delta_B>
and the seed-dependent part |Delta_A - Delta_B|^2 / 2; the window FIDs.

Outputs: $VMG_RUNS/edm2-64-xs-2e27/cfg-regimes/{gram/level<KK>.npz, sweep/<run>-<window>.json, summary.json}.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.experiments import ens_edm2, seedpair_in64  # noqa: E402
from vmg import runs  # noqa: E402
from vmg.models import _edm  # noqa: E402
from vmg.models.edm2 import EDMHeun, EDMPair  # noqa: E402

KIMG, EMA = 134217, "0.100"                       # 2^27 training images
BRANCHES = {"DAc": ("DA", True), "DAu": ("DA", False), "DBc": ("DB", True), "DBu": ("DB", False),
            "Ac": ("A", True), "Bc": ("B", True), "XLc": ("XL", True), "Lc": ("L", True)}   # name: (network, label)
W = 2.0
WINDOWS = {"none": None, "10-80": (10, 80), "2.9-10": (2.9, 10), "0.9-2.9": (0.9, 2.9), "0.28-0.9": (0.28, 0.9),
           "0-0.28": (0, 0.28)}                   # guidance where lo < sigma <= hi
POINTS = [(run, win) for run in ("DA", "DB") for win in WINDOWS]
N_GRAM, N_SAMPLES = 4096, 10000


def root():
    return runs.root() / "edm2-64-xs-2e27"


def out_dir(*sub):
    d = root().joinpath("cfg-regimes", *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def ckpt(net):
    """A released network (relative to $VMG_WEIGHTS) or one of our runs (absolute)."""
    if net in ("XL", "L"):
        return ens_edm2.MODELS[net]
    return str(root() / "train" / net / f"network-snapshot-{KIMG:07d}-{EMA}.pkl")


def atomic(path, write):
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    write(tmp)
    os.replace(tmp, path)


# --- the Gram of every branch -----------------------------------------------------------------------------------
def calibration(n):
    """uint8 [n, 3, 64, 64] and labels of n training images, drawn without replacement with seed 123."""
    m = len(np.load(seedpair_in64.zip_path().with_name("labels.npy")))
    return seedpair_in64.zip_images(np.random.default_rng(123).choice(m, n, replace=False))


@torch.no_grad()
def gram(a):
    dev, names = a.device, a.branches.split(",")
    sig = EDMHeun().levels()["sigma"]
    todo = [k for k in range(len(sig))
            if k % a.shards == a.shard and not (out_dir("gram") / f"level{k:02d}.npz").exists()]
    if not todo:
        print("every level of this shard is done")
        return
    u8, lab = calibration(a.n)
    x0 = torch.from_numpy(u8).to(dev).float() / 127.5 - 1
    onehot = torch.nn.functional.one_hot(torch.from_numpy(lab).to(dev), 1000).float()
    xbar = x0.mean(0, keepdim=True)
    nets = dict.fromkeys(BRANCHES[b][0] for b in names)
    for net in nets:
        if net in ("XL", "L"):                         # NVIDIA's, fetched on first use
            _edm.fetch(ckpt(net))
        nets[net] = _edm.edm_pickle(ckpt(net), dev)[0]
    R = (x0 - xbar).flatten(1).double()
    for k in todo:
        t0, s = time.time(), float(sig[k])
        eps = torch.randn(x0.shape, generator=torch.Generator(device=dev).manual_seed(1000 + k), device=dev)
        G, H = np.zeros((len(names), len(names))), np.zeros(len(names))
        for b0 in range(0, a.n, a.batch):
            x, c, r = x0[b0:b0 + a.batch], onehot[b0:b0 + a.batch], R[b0:b0 + a.batch]
            xt, t = x + s * eps[b0:b0 + a.batch], torch.full((len(x),), s, device=dev)
            P = torch.stack([nets[BRANCHES[b][0]](xt, t, c if BRANCHES[b][1] else torch.zeros_like(c)).float()
                             for b in names])
            P = (P - xbar).flatten(2).double()
            G += torch.einsum("kbd,lbd->kl", P, P).cpu().numpy()
            H += torch.einsum("kbd,bd->k", P, r).cpu().numpy()
        atomic(out_dir("gram") / f"level{k:02d}.npz",
               lambda p: np.savez(p, G=G, H=H, X2=float((R ** 2).sum()), n=a.n, sigma=s, names=np.array(names)))
        print(f"level {k:2d} sigma {s:8.4f}: {time.time() - t0:.0f}s", flush=True)


# --- windowed guidance ------------------------------------------------------------------------------------------
class DropoutCFG(EDMHeun):
    """One label-dropout network as the CFG pair of EDMHeun.heun: D1 with the class label, D0 with zero labels."""

    decode = EDMPair.decode

    def __init__(self, net, dev):
        self.net, self.encoder = _edm.edm_pickle(ckpt(net), dev)
        self.device = dev

    def pair(self, x, t, labels, released=False, ref=True):
        return self.net(x, t, labels), (self.net(x, t, torch.zeros_like(labels)) if ref else None)

    def initial(self, seeds):
        """generate_images.py: per seed, the noise and then the class, on this pair's device."""
        rnd = _edm.StackedRandomGenerator(self.device, seeds)
        noise = rnd.randn([len(seeds), 3, 64, 64], device=self.device)
        return noise, torch.eye(1000, device=self.device)[rnd.randint(1000, size=[len(seeds)], device=self.device)]


def weights(win):
    """The weight of each of the 32 levels: W inside the window, 1 elsewhere."""
    return np.ones(32) if WINDOWS[win] is None else ens_edm2.parse_w("interval:{:g}:{:g}:{:g}".format(W, *WINDOWS[win]))


def samples(pair, win, n, batch):
    """uint8 batches of seeds 0..n-1 sampled with the window's weights."""
    ws = weights(win)
    for b0 in range(0, n, batch):
        noise, labels = pair.initial(list(range(b0, min(n, b0 + batch))))
        yield pair.decode(pair.heun(noise, labels, ws))


def sweep(a):
    from vmg.metrics.common import frechet_stats
    from vmg.metrics.image import FD_NAME, Features, detectors, nvidia_stats
    run, win = POINTS[a.point]
    path = out_dir("sweep") / f"{run}-{win}.json"
    if path.exists():
        print(f"{path} exists")
        return
    t0, dev = time.time(), a.device
    _edm.fetch("edm2/dataset-refs/img64.pkl")
    acc = Features(detectors(dev))
    for u8 in samples(DropoutCFG(run, dev), win, a.n, a.batch):
        acc.update(u8)
    feats, ref = acc.numpy(), nvidia_stats(64)
    res = dict(run=run, window=win, w=W if WINDOWS[win] else 1.0, n=a.n, secs=round(time.time() - t0),
               **{FD_NAME[d]: frechet_stats(f, *ref[FD_NAME[d]]) for d, f in feats.items()})
    atomic(path, lambda p: p.write_text(json.dumps(res, indent=1)))
    print(json.dumps(res))


# --- analysis ---------------------------------------------------------------------------------------------------
def level_stats(z):
    """The per-level quantities of one Gram file (per image, summed over pixels)."""
    G, H, X2, n = z["G"], z["H"], float(z["X2"]), int(z["n"])
    i = {str(b): k for k, b in enumerate(z["names"])}
    d2 = lambda p, q: (G[i[p], i[p]] + G[i[q], i[q]] - 2 * G[i[p], i[q]]) / n
    dot = lambda p, q, r, s: (G[i[p], i[r]] - G[i[p], i[s]] - G[i[q], i[r]] + G[i[q], i[s]]) / n   # <p - q, r - s>
    mse = lambda p: (G[i[p], i[p]] - 2 * H[i[p]] + X2) / n
    out = dict(sigma=float(z["sigma"]), delta2=[d2(f"{r}c", f"{r}u") for r in ("DA", "DB")],
               err=[d2(f"{r}c", "XLc") for r in ("DA", "DB")], gain=[mse(f"{r}u") - mse(f"{r}c") for r in ("DA", "DB")],
               systematic=dot("DAc", "DAu", "DBc", "DBu"), dropout_pair=d2("DAc", "DBc"))
    out["seed"] = (sum(out["delta2"]) - 2 * out["systematic"]) / 2
    if "Ac" in i and "Bc" in i:
        out["cond_pair"] = d2("Ac", "Bc")
    if "Lc" in i:
        out["xl_bound"] = mse("XLc") - mse("Lc")
    return out


def steps_of(lo, hi):
    """The 1-based steps of the grid whose own level lies in (lo, hi]."""
    st = [k + 1 for k, s in enumerate(EDMHeun().levels()["sigma"]) if lo < s <= hi]
    return f"{st[0]}-{st[-1]}"


def analyze(a):
    files = sorted(out_dir("gram").glob("level*.npz"))
    assert len(files) == 32, f"{len(files)} of 32 Gram levels in {out_dir('gram')}"
    lv = sorted((level_stats(np.load(f)) for f in files), key=lambda r: -r["sigma"])
    L = {k: np.array([r[k] for r in lv]) for k in lv[0]}
    fid = {}
    for run in ("DA", "DB"):
        for win in WINDOWS:
            p = out_dir("sweep") / f"{run}-{win}.json"
            if p.exists():
                fid.setdefault(run, {})[win] = json.loads(p.read_text())["fid"]
    S = dict(n_images=int(np.load(files[0])["n"]), w=W, windows={k: v for k, v in WINDOWS.items() if v},
             steps={k: steps_of(*v) for k, v in WINDOWS.items() if v}, fid=fid,
             levels={k: v.tolist() for k, v in L.items()})
    (out_dir() / "summary.json").write_text(json.dumps(S, indent=1))

    sig, D2, E = L["sigma"], L["delta2"].mean(1), L["err"].mean(1)
    at = lambda s0: int(np.abs(np.log(sig / s0)).argmin())
    lines = [f"{'sigma':>7} {'|Delta|^2':>9} {'err vs XL':>9} {'seed part':>9} {'systematic':>10} {'label gain':>10}"]
    for k in range(len(sig)):
        lines.append(f"{sig[k]:7.3f} {D2[k]:9.2f} {E[k]:9.2f} {L['seed'][k]:6.2f} {100 * L['seed'][k] / D2[k]:2.0f}% "
                     f"{L['systematic'][k]:10.2f} {L['gain'][k].mean():10.2f}")
    k, j, m = at(80), at(2.9), at(0.28)
    sys_k, gain_k = L["systematic"][k], L["gain"][k]
    lines += [
        f"sigma {sig[k]:.3g}: |Delta|^2 / error {D2[k] / E[k]:.1f}; systematic part {100 * sys_k / D2[k]:.1f}% of "
        f"|Delta|^2, {100 * abs(gain_k.mean() - sys_k) / gain_k.mean():.1f}% from the label's MSE gain "
        f"({gain_k[0]:.1f}, {gain_k[1]:.1f})",
        f"sigma {sig[j]:.3g}: conditional error vs XL {E[j]:.1f}; "
        f"|D_DA(x_t,c) - D_DB(x_t,c)|^2 {L['dropout_pair'][j]:.1f}"
        + (f", |D_A(x_t,c) - D_B(x_t,c)|^2 {L['cond_pair'][j]:.1f}" if "cond_pair" in L else "")
        + (f"; error of XL >= MSE_XL - MSE_L = {L['xl_bound'][j]:.1f}" if "xl_bound" in L else ""),
        f"seed-dependent share of |Delta|^2: {100 * L['seed'][j] / D2[j]:.0f}% at sigma {sig[j]:.3g}, "
        f"{100 * L['seed'][m] / D2[m]:.0f}% at {sig[m]:.3g}",
        f"windows (w = {W:g}), steps: "
        + ", ".join(f"({lo:g}, {hi:g}] {S['steps'][w]}" for w, (lo, hi) in S["windows"].items())]
    for run, f in fid.items():
        change = {w: f" ({100 * (v / f['none'] - 1):+.1f}%)" if w != "none" and "none" in f else ""
                  for w, v in f.items()}
        lines.append(f"FID {run}: " + ", ".join(f"{w} {v:.2f}{change[w]}" for w, v in f.items()))
    (out_dir() / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# --- test -------------------------------------------------------------------------------------------------------
def tiny_nets():
    """Random EDM2 networks of the ImageNet-64 shape, small enough for a CPU, written where ckpt() looks."""
    import pickle
    _edm.edm_setup()
    from training.networks_edm2 import Precond
    for k, net in enumerate(dict.fromkeys(n for n, _ in BRANCHES.values())):
        torch.manual_seed(k)
        p = Path(os.environ["VMG_WEIGHTS"]) / ckpt(net) if net in ("XL", "L") else Path(ckpt(net))
        p.parent.mkdir(parents=True, exist_ok=True)
        ema = Precond(64, 3, 1000, model_channels=16, channel_mult=[1, 1], num_blocks=1, attn_resolutions=[]).eval()
        for name, w in ema.named_parameters():      # EDM2 starts at zero gains: every branch would be c_skip x
            if name.endswith("gain"):
                w.data.fill_(1.0)
        pickle.dump(dict(ema=ema), open(p, "wb"))


def tiny_zip(n):
    """An img64.zip of n random images in seedpair_in64's layout, with its labels.npy."""
    import io
    import zipfile
    import PIL.Image
    rng = np.random.default_rng(0)
    zp = seedpair_in64.zip_path()
    zp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, "w") as z:
        for r in range(n):
            b = io.BytesIO()
            PIL.Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(b, format="png")
            z.writestr(f"{r:08d}"[:5] + f"/img{r:08d}.png", b.getvalue())
    np.save(zp.with_name("labels.npy"), rng.integers(0, 1000, n).astype(np.int16))


@torch.no_grad()
def test():
    """On random networks: the Gram's sums against direct per-branch MSEs and distances, the sampler against
    edm_sampler, the windows, and analyze() on the Gram with made-up FIDs."""
    tmp = tempfile.mkdtemp(prefix="cfg-regimes-test-")
    os.environ["VMG_RUNS"] = tmp
    os.environ["VMG_WEIGHTS"] = str(Path(tmp) / "_weights")
    tiny_nets()
    tiny_zip(24)
    gram(argparse.Namespace(branches=",".join(BRANCHES), n=12, batch=5, shard=0, shards=1, device="cpu"))
    z = np.load(out_dir("gram") / "level15.npz")
    st, s = level_stats(z), float(z["sigma"])
    u8, lab = calibration(12)
    x0 = torch.from_numpy(u8).float() / 127.5 - 1
    c = torch.nn.functional.one_hot(torch.from_numpy(lab), 1000).float()
    xt = x0 + s * torch.randn(x0.shape, generator=torch.Generator().manual_seed(1015))
    net = _edm.edm_pickle(ckpt("DA"), "cpu")[0]
    t = torch.full((12,), s)
    dc, du = net(xt, t, c).double(), net(xt, t, torch.zeros_like(c)).double()
    direct = dict(delta2=float(((dc - du) ** 2).sum() / 12),
                  gain=float((((du - x0) ** 2).sum() - ((dc - x0) ** 2).sum()) / 12))
    for key, v in direct.items():
        assert np.isclose(st[key][0], v, rtol=1e-6), (key, st[key][0], v)
    assert np.allclose(z["G"], z["G"].T) and np.linalg.eigvalsh(z["G"]).min() > -1e-6 * np.trace(z["G"])

    gen = _edm.import_file("vmg_edm2_generate_images", _edm.src("edm2") / "generate_images.py")
    pair = DropoutCFG("DA", "cpu")
    noise, labels = pair.initial([0, 1, 2])
    theirs = gen.edm_sampler(net, noise, labels, gnet=lambda x, t, y: net(x, t, torch.zeros_like(y)), guidance=W)
    assert torch.equal(theirs, pair.heun(noise, labels, np.full(32, W))), "sampler differs from edm_sampler"
    assert torch.equal(gen.edm_sampler(net, noise, labels), pair.heun(noise, labels, weights("none")))
    ws = {w: weights(w) for w in WINDOWS}
    assert (sum(ws[w] - 1 for w in WINDOWS) == W - 1).all(), "the windows do not tile the grid"
    ims = {w: next(samples(pair, w, 3, 3)) for w in ("none", "0.9-2.9")}
    assert ims["none"].dtype == torch.uint8 and ims["none"].shape == (3, 3, 64, 64)
    assert not torch.equal(ims["none"], ims["0.9-2.9"])

    for k, (run, win) in enumerate(POINTS):
        (out_dir("sweep") / f"{run}-{win}.json").write_text(json.dumps(dict(fid=8.0 + 0.1 * k)))
    analyze(argparse.Namespace())
    shutil.rmtree(tmp)
    print("test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["gram", "sweep", "analyze", "test"])
    ap.add_argument("--branches", default=",".join(BRANCHES), help="gram: the branches, names of BRANCHES")
    ap.add_argument("--n", type=int, help=f"gram: images (default {N_GRAM}); sweep: samples (default {N_SAMPLES})")
    ap.add_argument("--batch", type=int, help="images per network call (default 128 for gram, 500 for sweep)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--point", type=int, help=f"sweep: index into POINTS (0..{len(POINTS) - 1})")
    a = ap.parse_args()
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.stage == "gram":
        a.n, a.batch = a.n or N_GRAM, a.batch or 128
        gram(a)
    elif a.stage == "sweep":
        a.n, a.batch = a.n or N_SAMPLES, a.batch or 500
        sweep(a)
    elif a.stage == "analyze":
        analyze(a)
    else:
        test()


if __name__ == "__main__":
    main()
