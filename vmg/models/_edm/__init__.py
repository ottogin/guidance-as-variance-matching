"""Third-party code and checkpoints behind the ImageNet and CIFAR-10 adapters (edm2.py, cfg256.py).

Nothing is vendored (EDM2 and EDM are CC BY-NC-SA 4.0, DiT CC BY-NC 4.0): every project is an external clone
at a pinned commit, put on sys.path at run time. NVIDIA's EDM2 and EDM pickles carry their own network source
(torch_utils.persistence) and unpickle with the NVlabs/edm2 clone on sys.path (its dnnlib and torch_utils,
which also load EDM v1's CIFAR-10 pickles); the ImageNet-256 models are state dicts whose classes come from
each project's own repository.

    VMG_EDM2      a clone of github.com/NVlabs/edm2 at REPOS["edm2"]    (required; vmg/metrics shares it)
    VMG_SRC       folder with the clones DiT/ SiT/ U-ViT/ REPA/ MaskDiT/  (default $VMG_RUNS/_src)
    VMG_WEIGHTS   checkpoints at the relative paths below; EDM2/EDM pickles mirror NVIDIA's CDN layout

    python -m vmg.models._edm clone              # git clone the six repositories at their pinned commits
    python -m vmg.models._edm fetch [path ...]   # download checkpoints and NVIDIA's ImageNet reference statistics
                                                 # (default: every one), sizes checked
    python -m vmg.models._edm rebuild [path ...] # an unreleased post-hoc EMA (posthoc-rebuilt/): download its run's
                                                 # raw snapshots, then edm2's reconstruct_phema.py (default: every one)
"""
import importlib.util
import os
import pickle
import subprocess
import sys
import urllib.request
from pathlib import Path

from ... import runs
from .. import hub

REPOS = {   # name -> (url, commit)
    "edm2": ("https://github.com/NVlabs/edm2", "4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc"),
    "DiT": ("https://github.com/facebookresearch/DiT", "ed81ce2229091fd4ecc9a223645f95cf379d582b"),
    "SiT": ("https://github.com/willisma/SiT", "cbde832a40b153ccc79603412409da9c9b0c568c"),
    "U-ViT": ("https://github.com/baofff/U-ViT", "ce551708dc9cde9818d2af7d84dfadfeb7bd9034"),
    "REPA": ("https://github.com/sihyun-yu/REPA", "67f714503e3892f993844aab088ffc5791c92613"),
    "MaskDiT": ("https://github.com/Anima-Lab/MaskDiT", "73cfb96ca9c6d1d78ab567ffeea099f96692a6e4"),
}
CDN = "https://nvlabs-fi-cdn.nvidia.com"


def posthoc(name):
    """An EDM2 post-hoc EMA reconstruction, e.g. edm2-img512-s-2147483-0.070."""
    return f"edm2/posthoc-reconstructions/{name}.pkl"


def raw(name):
    """An EDM2 raw training snapshot (EMA 0.050 or 0.100 tracked during training)."""
    return f"edm2/raw-snapshots/{name.rsplit('-', 2)[0]}/{name}.pkl"


def rebuilt(name):
    """An EDM2 post-hoc EMA that NVIDIA used but did not publish (e.g. ImageNet-64's CFG pair), rebuilt from the
    raw snapshots of its run (fetch() says how)."""
    return f"edm2/posthoc-rebuilt/{name}.pkl"


SNAPSHOT_KIMG = 2 ** 25 / 1000          # NVIDIA's raw snapshots: every 32Mi training images, EMA 0.050 and 0.100


def snapshots(run, kimg):
    """The released raw snapshots of one run up to `kimg` (what reconstruct_phema.py needs for an EMA at kimg)."""
    return [raw(f"{run}-{int(i * SNAPSHOT_KIMG):07d}-{s}") for i in range(1, int((kimg + 1) // SNAPSHOT_KIMG) + 1)
            for s in ("0.050", "0.100")]


def rebuild(rel):
    """Rebuild an unreleased post-hoc EMA (rebuilt()) from its run's raw snapshots, as NVIDIA's edm2 README does:
    download every snapshot up to its training time (30-160 GB a run), then reconstruct_phema.py."""
    dst = Path(os.environ.get("VMG_WEIGHTS", runs.root() / "_weights")) / rel
    if dst.exists():
        return dst
    run, kimg, std = Path(rel).stem.rsplit("-", 2)
    snaps = snapshots(run, int(kimg))
    print(f"{rel}: {len(snaps)} raw snapshots of {run}")
    for s in snaps:
        fetch(s)
    w = dst.parents[1]
    subprocess.run([sys.executable, str(src("edm2") / "reconstruct_phema.py"), f"--indir={w}/raw-snapshots/{run}",
                    f"--outdir={dst.parent}", f"--outprefix={run}", f"--outstd={std}", f"--outkimg={int(kimg)}"],
                   check=True, cwd=src("edm2"))
    assert dst.exists(), f"reconstruct_phema.py did not write {dst}"
    return dst


def edm1(name):
    """An EDM (v1) pretrained pickle, e.g. edm-cifar10-32x32-cond-vp."""
    return f"edm/pretrained/{name}.pkl"


# checkpoints that are not on NVIDIA's CDN: relative path -> release URL ("gdrive:<id>" for Google Drive)
OTHER = {
    "dit/DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "sit/SiT-XL-2-256.pt": "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt"
                           "?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0",
    "uvit/imagenet256_uvit_large.pth": "gdrive:1w7T1hiwKODgkYyMH9Nc9JNUThbxFZgs3",
    "uvit/imagenet256_uvit_huge.pth": "gdrive:13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u",
    "maskdit/imagenet256_with_guidance.pt": "https://huggingface.co/hzzheng/MaskDiT/resolve/main/checkpoints/"
                                            "imagenet256_with_guidance.pt",
    "repa/repa-sit-xl2-256-4m.pt": "https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt"
                                   "?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0",
}
SD_VAE = "stabilityai/sd-vae-ft-mse"     # the VAE of every ImageNet latent model here (EDM2-512 and the 256 zoo)


def _clone_dir(name):
    if name == "edm2":
        assert os.environ.get("VMG_EDM2"), ("set VMG_EDM2 to a clone of https://github.com/NVlabs/edm2 at commit "
                                            f"{REPOS['edm2'][1]} (`python -m vmg.models._edm clone` makes one there)")
        return Path(os.environ["VMG_EDM2"])
    return Path(os.environ.get("VMG_SRC", runs.root() / "_src")) / name


def src(name):
    d = _clone_dir(name)
    assert d.exists(), f"{d} missing: run `python -m vmg.models._edm clone` (or point VMG_EDM2 / VMG_SRC at clones)"
    return d


def weight(rel):
    p = Path(os.environ.get("VMG_WEIGHTS", runs.root() / "_weights")) / rel
    how = "rebuild" if "/posthoc-rebuilt/" in rel else "fetch"
    assert p.exists(), f"{p} missing: run `python -m vmg.models._edm {how} {rel}`"
    return p


def url(rel):
    return OTHER.get(rel) or f"{CDN}/{rel}"


def import_file(name, path):
    """Import one source file under a unique module name (DiT's and SiT's are both called models.py)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def on_path(d):
    d = str(d)
    if d not in sys.path:
        sys.path.insert(0, d)


def edm_setup():
    """Put the edm2 clone (dnnlib, torch_utils, training) on sys.path; keep dnnlib's cache off $HOME."""
    os.environ.setdefault("DNNLIB_CACHE_DIR", str(runs.root() / "_cache" / "dnnlib"))
    on_path(src("edm2"))


_PICKLES = {}


def edm_pickle(rel, device="cuda"):
    """(net, encoder) of an EDM2 or EDM pickle, loaded once per device. encoder is None for EDM (v1)."""
    key = (rel, str(device))
    if key not in _PICKLES:
        edm_setup()
        with open(weight(rel), "rb") as f:
            data = pickle.load(f)
        enc = data.get("encoder")
        if enc is not None and hasattr(enc, "vae_name"):
            enc.vae_name = hub.path(SD_VAE)            # a local mirror when VMG_WEIGHTS has one
        _PICKLES[key] = (data["ema"].to(device).eval().requires_grad_(False), enc)
    return _PICKLES[key]


class StackedRandomGenerator:
    """edm2/generate_images.py's per-seed generator: seed i gives image i's noise and then its class."""

    def __init__(self, device, seeds):
        import torch
        self.generators = [torch.Generator(device).manual_seed(int(s) % (1 << 32)) for s in seeds]

    def randn(self, size, **kw):
        import torch
        return torch.stack([torch.randn(size[1:], generator=g, **kw) for g in self.generators])

    def randint(self, *args, size, **kw):
        import torch
        return torch.stack([torch.randint(*args, size=size[1:], generator=g, **kw) for g in self.generators])


# --- fetching ---------------------------------------------------------------------------------------------
def clone():
    for name, (u, commit) in REPOS.items():
        d = _clone_dir(name)
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            subprocess.run(["git", "clone", "-q", u, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", commit], check=True)
        print(f"{name}: {commit[:12]}")


def fetch(rel):
    dst = Path(os.environ.get("VMG_WEIGHTS", runs.root() / "_weights")) / rel
    if dst.exists():
        return dst
    if "/posthoc-rebuilt/" in rel:                       # not on the CDN: reconstructed locally
        print(f"{rel}: not released; rebuild it with `python -m vmg.models._edm rebuild {rel}`")
        return None
    if rel.endswith("-ema.pt"):                          # MaskDiT: the EMA half of the released checkpoint
        return maskdit_ema(fetch(rel.replace("-ema.pt", ".pt")))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{dst}.{os.getpid()}.tmp")
    u = url(rel)
    if u.startswith("gdrive:"):                          # past Drive's virus-scan page for large files
        u = f"https://drive.usercontent.google.com/download?id={u[len('gdrive:'):]}&export=download&confirm=t"
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "vmg"}), timeout=600) as r:
        assert "text/html" not in r.headers.get("Content-Type", ""), f"{rel}: got a web page, not the file ({u})"
        want = int(r.headers.get("Content-Length", -1))
        with open(tmp, "wb") as f:
            while chunk := r.read(1 << 24):
                f.write(chunk)
    got = tmp.stat().st_size
    assert want in (-1, got), f"{rel}: wrote {got} of {want} bytes (a short write: retry)"
    os.replace(tmp, dst)
    print(f"{rel}: {got / 1e6:.0f} MB")
    return dst


def maskdit_ema(full):
    """MaskDiT ships model + EMA + optimizer (11.7 GB) with torch.compile's key prefix; keep the EMA, once."""
    import torch
    small = Path(str(full).replace(".pt", "-ema.pt"))
    if not small.exists():
        sd = torch.load(full, map_location="cpu", weights_only=False)["ema"]
        tmp = f"{small}.{os.getpid()}.tmp"
        torch.save({k.removeprefix("_orig_mod."): v for k, v in sd.items()}, tmp)
        os.replace(tmp, small)
    return small


# NVIDIA's FID / FD_DINOv2 statistics of the ImageNet training sets (vmg/metrics/image.py nvidia_stats)
DATASET_REFS = ("edm2/dataset-refs/img512.pkl", "edm2/dataset-refs/img64.pkl")


def every_checkpoint():
    from .. import cfg256, edm2
    return sorted({*edm2.checkpoints(), *cfg256.checkpoints(), *DATASET_REFS})


def main():
    cmd, *args = sys.argv[1:] or ["help"]
    if cmd == "clone":
        clone()
    elif cmd == "fetch":
        for rel in args or every_checkpoint():
            fetch(rel)
    elif cmd == "rebuild":
        for rel in args or [c for c in every_checkpoint() if "/posthoc-rebuilt/" in c]:
            rebuild(rel)
    else:
        print(__doc__)
