"""The paper's image metrics: FID, FD_DINOv2, precision / recall / density / coverage, and CLIP score.

    feats = Features(detectors())          # Inception-v3 (2048-d) and DINOv2 ViT-L/14 (1024-d)
    feats.update(u8)                       # uint8 [B, 3, S, S] at the metric size
    scores(feats_fake, feats_real)         # -> dict(fid, fd_dinov2, precision_inception, ..., coverage_dinov2)

The two detectors are EDM2's (calculate_metrics.py, Karras et al. 2024): the StyleGAN3 Inception pickle, and
DINOv2 from torch.hub with a bicubic antialiased resize to 224 and ImageNet normalisation. The Inception pickle
needs NVIDIA's torch_utils to unpickle; it is not copied here (EDM2 is CC BY-NC-SA), so set VMG_EDM2 to a clone
of github.com/NVlabs/edm2 (the paper used commit 4bf8162). Every
image, real or generated, reaches them at one metric size (256 px for COCO and MJHQ): a generated image by one
torch bicubic antialiased resize from its render size, a real one by a centre crop to its own short side and
one PIL bicubic resize, so a 375 px photograph is never upsampled to 480 first. Precision and recall use k = 3
on the same features (vmg/metrics/prdc.py). CLIP score is the mean cosine of openai/clip-vit-large-patch14
image and text embeddings.

ImageNet-512 (the EDM2-S sweep of app:tuned-w, tab:edm2-fail) and ImageNet-64 (tab:edm2-fail) are scored
at the native 512 or 64 px, with no resize before the detectors. FID and FD_DINOv2
are against NVIDIA's training statistics (edm2/dataset-refs/img512.pkl or img64.pkl, a mean and a covariance per
detector, passed as each space's fd_stats). Precision / recall / density / coverage are against 10,000 held-out
validation images (vmg/data/imagenet.py heldout_rows), centre-cropped at 512 or 64 as EDM2's dataset_tool does. Generated row r is seed r, and its CLIP prompt is "a photo of a <class>" for the class that
seed draws, with torchvision's ImageNet class names.
"""
import functools
import os
import socket
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .. import runs
from .common import frechet, scores as _scores  # noqa: F401  (frechet: re-exported for the checks)

INCEPTION_URL = ("https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/"
                 "inception-2015-12-05.pkl")
DINOV2 = ("facebookresearch/dinov2:main", "dinov2_vitl14")
CLIP_ID = os.environ.get("VMG_CLIP", "openai/clip-vit-large-patch14")
METRIC_SIZE = 256
DETECTORS = ("inception", "dinov2")             # FID and FD_DINOv2, in this order
FD_NAME = {"inception": "fid", "dinov2": "fd_dinov2"}
PRDC_K = 3


def cache_dir():
    d = Path(os.environ.get("VMG_CACHE", runs.root() / "_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def nvlabs_path(var="VMG_EDM2"):
    """Put an NVlabs clone (EDM2 by default) on sys.path; its pickles unpickle only with its torch_utils."""
    import sys
    d = os.environ.get(var)
    assert d and Path(d, "torch_utils").is_dir(), (f"set {var} to a clone of github.com/NVlabs/"
                                                    f"{'edm2' if var == 'VMG_EDM2' else 'edm'} (see reproduce/envs)")
    if d not in sys.path:
        sys.path.append(d)


class Inception:
    feature_dim = 2048

    def __init__(self, device="cuda"):
        nvlabs_path()
        path = cache_dir() / "inception-2015-12-05.pkl"
        if not path.exists():
            tmp = f"{path}.{socket.gethostname()}-{os.getpid()}.tmp"      # parallel jobs may download at once
            urllib.request.urlretrieve(INCEPTION_URL, tmp)
            os.replace(tmp, path)
        with open(path, "rb") as f:
            self.model = pickle.load(f).to(device).eval()

    @torch.no_grad()
    def __call__(self, u8):
        return self.model(u8.to(torch.uint8), return_features=True)


class DINOv2:
    feature_dim = 1024

    def __init__(self, device="cuda"):
        import warnings
        warnings.filterwarnings("ignore", "xFormers is not available")
        torch.hub.set_dir(os.environ.get("TORCH_HUB", str(cache_dir() / "torch_hub")))
        self.model = torch.hub.load(*DINOV2, trust_repo=True, verbose=False, skip_validation=True)
        self.model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, u8):
        x = torch.nn.functional.interpolate(u8.to(torch.float32), size=(224, 224), mode="bicubic", antialias=True)
        x = x / 255
        x = (x - x.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)) / x.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        return self.model(x)


def detectors(device="cuda"):
    return {"inception": Inception(device), "dinov2": DINOv2(device)}


def to_metric(u8, size=METRIC_SIZE):
    """A generated batch (uint8 [B, 3, H, W]) at the metric size: one torch bicubic antialiased resize."""
    if u8.shape[-2:] == (size, size):
        return u8
    x = torch.nn.functional.interpolate(u8.float(), size=(size, size), mode="bicubic", antialias=True)
    return x.clamp(0, 255).round().to(torch.uint8)


def real_to_metric(im, size=METRIC_SIZE):
    """A real PIL image at the metric size (uint8 [size, size, 3]): centre crop to its own short side, then
    one PIL bicubic resize."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if s != size:
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


class Features:
    """Every image's detector features (float32), kept for the k-NN metrics; the Frechet statistics are
    computed from them in float64."""

    def __init__(self, dets):
        self.dets = dets
        self.feats = {k: [] for k in dets}

    def update(self, u8):
        for k, det in self.dets.items():
            self.feats[k].append(det(u8).float().cpu().numpy())

    def numpy(self):
        return {k: np.concatenate(v) for k, v in self.feats.items()}


def spaces(model=None, dataset=None):
    """Inception (FID) and DINOv2 (FD_DINOv2), each with precision / recall / density / coverage. On ImageNet the
    Frechet distances are against NVIDIA's training statistics (fd_stats), the k-NN metrics against held-out val."""
    if dataset in IMAGENET:
        st = nvidia_stats(IMAGENET[dataset])
        return {d: dict(fd=FD_NAME[d], prdc=True, fd_stats=st[FD_NAME[d]]) for d in DETECTORS}
    return {d: dict(fd=FD_NAME[d], prdc=True) for d in DETECTORS}


def scores(fake, real, k=PRDC_K):
    """FID, FD_DINOv2 and precision / recall / density / coverage per detector."""
    out = _scores(fake, real, spaces(), k)
    out.update(n=len(fake[DETECTORS[0]]), prdc_k=k)
    return out


def _dataset(name):
    """(real PIL image of row i, prompts of rows) for the image evaluation sets."""
    from PIL import Image
    if name == "coco":
        from ..data import coco as D
        path = lambda i: D.root() / "val2014" / D.select()[i]["file_name"]       # noqa: E731
    elif name == "mjhq":
        from ..data import mjhq as D
        path = lambda i: D.root() / D.select()[i]["file_name"]                   # noqa: E731
    else:
        raise SystemExit(f"no image evaluation set {name!r}")
    return (lambda i: Image.open(path(i))), D.captions


BATCH = 64


def _batches(ids):
    for b in range(0, len(ids), BATCH):
        yield ids[b:b + BATCH]


_DETS = {}


def _detectors():
    if not _DETS:
        _DETS.update(detectors())
    return _DETS


def reference(model, dataset, ids):
    """The evaluation rows' own images at the metric size: their detector features."""
    if dataset in IMAGENET:
        return _imagenet_reference(dataset, ids)
    real, _ = _dataset(dataset)
    acc = Features(_detectors())
    for b in _batches(ids):
        acc.update(torch.from_numpy(np.stack([real_to_metric(real(i)) for i in b])).permute(0, 3, 1, 2).cuda())
        print(f"  reference {b[-1] - ids[0] + 1}/{len(ids)}", flush=True)
    return acc.numpy(), {}


def score_arm(model, dataset, ids, src, ref):
    """An arm's PNGs <row>.png: detector features (one torch resize to the metric size) and per-row CLIP score."""
    from PIL import Image
    files = [src / f"{i:05d}.png" for i in ids]
    missing = [f.name for f in files if not f.exists()]
    assert not missing, f"{len(missing)} of {len(files)} samples missing in {src}, e.g. {missing[:3]}"
    if dataset in IMAGENET:
        return _imagenet_arm(model, dataset, ids, src)
    _, prompts = _dataset(dataset)
    clip, acc, cos = ClipScore(), Features(_detectors()), []
    for b in _batches(ids):
        u8 = torch.from_numpy(np.stack([np.asarray(Image.open(src / f"{i:05d}.png").convert("RGB")) for i in b]))
        u8 = u8.permute(0, 3, 1, 2).cuda()
        acc.update(to_metric(u8))
        cos.append(clip(u8, prompts(b)))
        print(f"  {src.name} {b[-1] - ids[0] + 1}/{len(ids)}", flush=True)
    return acc.numpy(), {"clip": np.concatenate(cos)}


# --- ImageNet ------------------------------------------------------------------------------------------------
IMAGENET = {"imagenet512": 512, "imagenet64": 64}     # dataset -> resolution, which is also its metric size
CLASS_PROMPT = "a photo of a {}"


def imagenet_names():
    """The 1,000 ImageNet class names in index order (torchvision's)."""
    from torchvision.models import ResNet50_Weights
    return ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


@functools.lru_cache(maxsize=2)
def nvidia_stats(res):
    """{"fid": (mu, sigma), "fd_dinov2": (mu, sigma)} of NVIDIA's ImageNet-<res> training set (EDM2 dataset-refs),
    from $VMG_WEIGHTS/edm2/dataset-refs/img<res>.pkl (python -m vmg.models._edm fetch edm2/dataset-refs/img<res>.pkl)."""
    from ..models import _edm
    with open(_edm.weight(f"edm2/dataset-refs/img{res}.pkl"), "rb") as f:
        ref = pickle.load(f)
    return {m: (np.asarray(ref[m]["mu"], np.float64), np.asarray(ref[m]["sigma"], np.float64)) for m in FD_NAME.values()}


def _imagenet_reference(dataset, ids):
    """Held-out validation image i for evaluation row i (k-NN metrics), and NVIDIA's statistics (Frechet)."""
    from ..data import imagenet as D
    res = IMAGENET[dataset]
    rows = [D.heldout_rows(res=res)[i] for i in ids]
    acc = Features(_detectors())
    for b in _batches(list(range(len(rows)))):
        u8 = torch.from_numpy(D.real_u8([rows[j] for j in b], res)).permute(0, 3, 1, 2).cuda()
        acc.update(u8)
        print(f"  reference {b[-1] + 1}/{len(rows)}", flush=True)
    return acc.numpy(), {}


def _imagenet_arm(model, dataset, ids, src):
    from PIL import Image
    names, res = imagenet_names(), IMAGENET[dataset]
    clip, acc, cos = ClipScore(), Features(_detectors()), []
    for b in _batches(ids):
        u8 = torch.from_numpy(np.stack([np.asarray(Image.open(src / f"{i:05d}.png").convert("RGB")) for i in b]))
        u8 = u8.permute(0, 3, 1, 2).cuda()
        acc.update(to_metric(u8, res))
        cos.append(clip(u8, [CLASS_PROMPT.format(names[c]) for c in model.classes(b)]))
        print(f"  {src.name} {b[-1] - ids[0] + 1}/{len(ids)}", flush=True)
    return acc.numpy(), {"clip": np.concatenate(cos)}


class ClipScore:
    def __init__(self, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        from ..models import hub
        self.model = CLIPModel.from_pretrained(hub.path(CLIP_ID), torch_dtype=torch.float32).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(hub.path(CLIP_ID))
        self.device = device

    @torch.no_grad()
    def __call__(self, u8, prompts):
        """Per-image cosine between the image (uint8 [B, 3, H, W]) and its prompt."""
        pix = [x.permute(1, 2, 0).cpu().numpy() for x in u8]
        inp = self.proc(text=prompts, images=pix, return_tensors="pt", padding=True, truncation=True).to(self.device)
        o = self.model(**inp)
        ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        return (ie * te).sum(-1).float().cpu().numpy()
