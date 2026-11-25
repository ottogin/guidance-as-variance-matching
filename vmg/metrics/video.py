"""The paper's video and world-model metrics: FVD and k = 3 precision / recall / density / coverage in I3D features,
CLIP-T, Latte's C3D Inception Score, and Cosmos-Predict2.5's frame accuracy (PSNR, SSIM).

Detectors are StyleGAN-V's (Skorokhodov et al. 2022), the torchscripts Latte's eval_metrics.sh runs, downloaded on
first use and checked by md5:
  I3D   i3d(u8 [B, 3, T, H, W], rescale=True, resize=True, return_features=True) -> [B, 400]; it resizes to 224
  C3D   the UCF-101 classifier, 16 frames -> class probabilities [B, 101]; IS = StyleGAN-V's isv (10 splits after a
        RandomState(42) shuffle)
Frames reach them uint8 and uncompressed, exactly as the paper's runs fed them:
  svd        all 14 frames at 1024 x 576 (real: UCF-101 frames bicubic-resized like the conditioning image)
  latte      all 16 frames at 256^2 (real: Latte's training transform of the same row's clip)
  wan, cogvideox
             16 evenly spaced frames of the decoded MP4 (real: the clip read as the moments read it: the model's
             fps, frame count and size); CLIP-T is CLIP ViT-L/14, 100 x the mean cosine of 8 evenly spaced frames
             with the caption (77 tokens)
  cosmos     non-overlapping 16-frame windows from frame 1 of each rolled-out episode and of its recorded frames
             (FVD only: the windows vary in number per episode); PSNR and SSIM per episode over frames 1..N-1
Metric names: fvd; precision_i3d, recall_i3d, density_i3d, coverage_i3d; is_c3d; clip_t; psnr, ssim.
"""
import hashlib
import os
import socket
import urllib.request

import numpy as np
import torch

from .image import cache_dir

DET = {"i3d": ("https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1", "8d45542982dc92282a73cb852e3c53d4"),
       "c3d": ("https://www.dropbox.com/s/jxpu7avzdc9n97q/c3d_ucf101.pt?dl=1", "2fad867904985ce060bb9a746146362c")}
N_I3D, N_CLIP, WINDOW, BATCH = 16, 8, 16, 8
CLIP_ID = os.environ.get("VMG_CLIP", "openai/clip-vit-large-patch14")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


_NETS = {}


def detector(name):
    """A StyleGAN-V torchscript on the GPU ("i3d" or "c3d")."""
    if name not in _NETS:
        url, md5 = DET[name]
        path = cache_dir() / f"{name}_{'torchscript' if name == 'i3d' else 'ucf101'}.pt"
        if not path.exists():
            tmp = f"{path}.{socket.gethostname()}-{os.getpid()}.tmp"      # parallel jobs may download at once
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        assert _md5(path) == md5, f"{path}: md5 is not {md5}"
        _NETS[name] = torch.jit.load(str(path), map_location="cuda").eval()
    return _NETS[name]


@torch.no_grad()
def i3d(videos):
    """uint8 [B, T, H, W, 3] -> I3D features float64 [B, 400]."""
    x = torch.as_tensor(np.ascontiguousarray(videos)).cuda().permute(0, 4, 1, 2, 3).contiguous()
    return detector("i3d")(x, rescale=True, resize=True, return_features=True).double().cpu().numpy()


@torch.no_grad()
def c3d(videos):
    """uint8 [B, 16, H, W, 3] -> C3D class probabilities float64 [B, 101]."""
    assert videos.shape[1] == 16, "the C3D Inception Score is defined on 16 frames"
    x = torch.as_tensor(np.ascontiguousarray(videos)).cuda().permute(0, 4, 1, 2, 3).contiguous()
    return detector("c3d")(x).double().cpu().numpy()


def inception_score(probs, splits=10, seed=42):
    """StyleGAN-V's isv: shuffle with RandomState(42), then the mean over splits of exp(E KL(p(y|x) || p(y)))."""
    p = np.asarray(probs, np.float64).copy()
    np.random.RandomState(seed).shuffle(p)
    n = len(p)
    splits = min(splits, n)                      # a smoke-sized set (< 10 rows) leaves no split empty
    sc = []
    for i in range(splits):
        part = p[i * n // splits:(i + 1) * n // splits]
        sc.append(np.exp((part * (np.log(part) - np.log(part.mean(0, keepdims=True)))).sum(1).mean()))
    return float(np.mean(sc))


def pick(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


def windows(v, start=1):
    """Non-overlapping 16-frame windows from frame `start` (Cosmos's conditioning frame 0 is excluded)."""
    return [v[start + WINDOW * i:start + WINDOW * (i + 1)] for i in range((len(v) - start) // WINDOW)]


class ClipT:
    """100 x the mean cosine between CLIP ViT-L/14 embeddings of 8 frames and of the caption."""

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor

        from ..models import hub
        self.model = CLIPModel.from_pretrained(hub.path(CLIP_ID)).to("cuda").eval()
        self.proc = CLIPProcessor.from_pretrained(hub.path(CLIP_ID))

    @staticmethod
    def _emb(o):                                  # transformers 5 may return a model output from get_*_features
        if torch.is_tensor(o):
            return o
        for k in ("text_embeds", "image_embeds", "pooler_output"):
            if getattr(o, k, None) is not None:
                return getattr(o, k)
        return o[0]

    @torch.no_grad()
    def __call__(self, frames, caption):
        F = torch.nn.functional
        t = self.proc(text=[caption], return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")
        te = F.normalize(self._emb(self.model.get_text_features(**t)), dim=-1)[0]
        im = self.proc(images=list(frames), return_tensors="pt").to("cuda")
        ie = F.normalize(self._emb(self.model.get_image_features(**im)), dim=-1)
        return float((ie @ te).mean() * 100)


# --- what each model's rows are --------------------------------------------------------------------------------

def _key(model):
    k = model.spec.key
    assert k in ("svd", "latte", "wan", "cogvideox", "cosmos"), f"no video metrics for {k!r}"
    return k


def _real(model, dataset, i):
    """Row i of the real reference as uint8 [T, H, W, 3]."""
    k = _key(model)
    if k == "svd":
        from ..data import ucf101
        return np.stack([np.asarray(f) for f in model.frames(ucf101.svd_clip(i))])
    if k == "latte":
        from ..data import ucf101
        x = model.pixels(ucf101.latte_clip(i)[None])[0]                    # [16, 3, 256, 256] in [-1, 1]
        u8 = ((x * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(torch.uint8)   # sample_ddp.py's rounding
        return u8.permute(0, 2, 3, 1).cpu().numpy()
    if k in ("wan", "cogvideox"):
        from ..data import openvid
        x = openvid.read_clip("eval", i, model.FPS, model.NF, model.H, model.W)
        return ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()
    from ..data import bridge
    return model.episode(bridge.eval_episodes()[i])[2]


def _fake(model, src, i):
    """Row i of an arm as uint8 [T, H, W, 3]."""
    path = model.sample_file(src, i)
    assert path.exists(), f"missing sample {path}"
    if path.suffix == ".mp4":
        import av
        with av.open(str(path)) as c:
            return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])])
    z = np.load(path)
    return z["pred" if "pred" in z.files else "video"]


def _caption(model, i):
    from ..data import openvid
    return openvid.captions("eval", [i])[0]


def spaces(model, dataset):
    """I3D (FVD, and the k-NN metrics except for Cosmos's windows); for Latte also C3D, whose statistic is the
    Inception Score of the arm."""
    k = _key(model)
    if k == "cosmos":
        return {"i3d": dict(fd="fvd", prdc=False, rows="i3d_row")}       # 16-frame windows of each episode
    out = {"i3d": dict(fd="fvd", prdc=True)}
    if k == "latte":
        out["c3d"] = dict(fd=None, prdc=False, stats={"is_c3d": inception_score})
    return out


def _features(model, dataset, ids, get, src=None):
    """({space: features}, {per-row metric: values}) of the videos get(i), batched through the detectors."""
    k = _key(model)
    feats = {"i3d": []}
    if k == "latte":
        feats["c3d"] = []
    rows = {"clip_t": []} if k in ("wan", "cogvideox") else {}
    if k == "cosmos":
        feats["i3d_row"] = []
        rows = {"psnr": [], "ssim": []} if src is not None else {}
    clip = ClipT() if "clip_t" in rows else None
    buf = []

    def flush():
        if buf:
            v = np.stack(buf)
            feats["i3d"].append(i3d(v))
            if "c3d" in feats:
                feats["c3d"].append(c3d(v))
            buf.clear()

    for j, i in enumerate(ids):
        v = get(i)
        if k == "cosmos":
            for w in windows(v):
                buf.append(w)
                feats["i3d_row"].append(j)
                if len(buf) == BATCH:
                    flush()
            if src is not None:
                from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                gt = _real(model, dataset, i)
                assert v.shape == gt.shape, f"episode {i}: prediction {v.shape} vs recorded {gt.shape}"
                rows["psnr"].append(np.mean([peak_signal_noise_ratio(g, q, data_range=255) for q, g in zip(v[1:], gt[1:])]))
                rows["ssim"].append(np.mean([structural_similarity(g, q, channel_axis=-1, data_range=255)
                                             for q, g in zip(v[1:], gt[1:])]))
        else:
            if k in ("wan", "cogvideox"):
                assert v.shape == (model.NF, model.H, model.W, 3), f"row {i}: {v.shape}"
                rows["clip_t"].append(clip(v[pick(len(v), N_CLIP)], _caption(model, i)))
                v = v[pick(len(v), N_I3D)]
            buf.append(v)
            if len(buf) == BATCH:
                flush()
        if (j + 1) % 64 == 0 or j + 1 == len(ids):
            print(f"  {'real' if src is None else src.name} {j + 1}/{len(ids)}", flush=True)
    flush()
    out = {sp: np.concatenate(f) for sp, f in feats.items() if sp != "i3d_row"}
    if k == "cosmos":
        out["i3d_row"] = np.asarray(feats["i3d_row"], dtype=np.int64)    # each window's position in ids
    return out, {m: np.asarray(v, dtype=np.float64) for m, v in rows.items()}


def reference(model, dataset, ids):
    """The evaluation rows' own clips (Cosmos: the recorded episodes); CLIP-T of the real clips is kept too."""
    return _features(model, dataset, ids, lambda i: _real(model, dataset, i))


def score_arm(model, dataset, ids, src, ref):
    return _features(model, dataset, ids, lambda i: _fake(model, src, i), src=src)
