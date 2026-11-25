"""ImageNet validation images for the EDM2 and ImageNet-256 models (Fig 2's autoguidance and class-CFG pairs).

The validation split is held out from every model, so the residuals measure generalisation. The arrays live
in $VMG_IMAGENET (default $VMG_DATA/imagenet) and are built once from the official validation images
(a manual download from image-net.org: ILSVRC2012_img_val, one folder per synset):

    img64_val_u8.npy     uint8   [50000, 64, 64, 3]   ADM's centre crop (EDM2's center-crop-dhariwal) at 64
    img512_val_vae.npy   float32 [50000, 8, 64, 64]   EDM2's StabilityVAEEncoder.encode_pixels at 512: mean | std
    img256_val_vae.npy   float32 [50000, 8, 32, 32]   sd-vae-ft-mse at 256 on pixels in [-1, 1] (DiT's convention)
    img{64,256,512}_val_labels.npy                    class = rank of the sorted synset folder (dataset_tool.py's)

    python -m vmg.data.imagenet prep 64  --src /path/to/val       # CPU
    python -m vmg.data.imagenet prep 512 --src /path/to/val       # GPU (the VAE encoder)
    python -m vmg.data.imagenet prep 256 --src /path/to/val       # GPU

Row r is the r-th image of the sorted validation listing. Calibration takes 10 images per class (10,000),
drawn once with seed 0 and kept in listing order; the 50,000-row passes take every image. A latent x0 is one
draw from the VAE posterior, mean + std * eps with eps from the row's own generator, then the model's
whitening, as training fed it.

The metrics' real reference (heldout_rows) is 10 other images per class, disjoint from the calibration
subset, drawn with seed 1 from the rest; their pixels come from the JPEGs in $VMG_IMAGENET_VAL, through the
same centre crop at the metric resolution. A tree holding only those files works if $VMG_IMAGENET_VAL_LIST
names the full listing (the 50,000 relative paths in order), which fixes the indices and the classes.
"""
import argparse
import functools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import noise, runs

N_VAL, CLASSES = 50000, 1000
_EXTS = {".jpeg", ".jpg", ".png"}


def root():
    return Path(os.environ.get("VMG_IMAGENET", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "imagenet"))


@functools.lru_cache(maxsize=None)
def labels(res=64):
    """Row classes. Every prep writes the same listing's labels, so any resolution's file serves (prep 512 alone
    is enough for the EDM2-512 models)."""
    have = [root() / f"img{r}_val_labels.npy" for r in (res, 64, 512, 256)]
    p = next((f for f in have if f.exists()), have[0])
    return np.load(p).astype(np.int64)


def subset(lab, n=None, seed=0):
    """Deterministic class-stratified subset of row ids."""
    if n is None or n >= len(lab):
        return np.arange(len(lab))
    rng = np.random.default_rng(seed)
    per_class = max(1, n // int(lab.max() + 1))
    picks = [rng.choice(np.flatnonzero(lab == c), size=min(per_class, int((lab == c).sum())), replace=False)
             for c in np.unique(lab)]
    idx = np.concatenate(picks)
    rng.shuffle(idx)                      # truncating a class-ordered list would favour low classes
    return np.sort(idx[:n])


def calib_rows(n):
    return [int(i) for i in subset(labels(64), n, seed=0)]


@functools.lru_cache(maxsize=None)
def _array(name):
    return np.load(root() / name, mmap_mode="r")


def pixels(rows):
    """[B, 3, 64, 64] float32 in [-1, 1] (EDM2's StandardRGBEncoder)."""
    import torch
    u8 = np.ascontiguousarray(_array("img64_val_u8.npy")[list(rows)])
    return torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 127.5 - 1


def latents(res, rows, scale, bias, seed=0):
    """[B, 4, res/8, res/8]: one posterior draw per row, times `scale` plus `bias` (per channel or scalar)."""
    import torch
    raw = torch.from_numpy(np.ascontiguousarray(_array(f"img{res}_val_vae.npy")[list(rows)], dtype=np.float32))
    mean, std = raw[:, :4], raw[:, 4:]
    eps = torch.stack([torch.randn(mean.shape[1:], generator=noise.generator(seed, r, 1)) for r in rows])
    sc = torch.as_tensor(np.asarray(scale, np.float32)).reshape(1, -1, 1, 1)
    bi = torch.as_tensor(np.asarray(bias, np.float32)).reshape(1, -1, 1, 1)
    return (mean + eps * std) * sc + bi


def val_root():
    return Path(os.environ.get("VMG_IMAGENET_VAL", root() / "val"))


@functools.lru_cache(maxsize=None)
def val_listing():
    """The 50,000 validation images' relative paths in listing order, checked against the stored classes."""
    lst = os.environ.get("VMG_IMAGENET_VAL_LIST")
    rel = open(lst).read().split() if lst else [str(Path(p).relative_to(val_root())) for p in listing(val_root())[0]]
    syn = {s: i for i, s in enumerate(sorted({r.split("/")[0] for r in rel}))}
    assert len(rel) == N_VAL and np.array_equal([syn[r.split("/")[0]] for r in rel], labels(64)), \
        "the validation listing does not match the stored classes"
    return rel


@functools.lru_cache(maxsize=None)
def heldout_rows(n=10000, fit_n=10000, seed=1, res=512):
    """The metrics' real images: n class-stratified val images disjoint from the calibration subset."""
    lab = labels(res)
    rest = np.setdiff1d(np.arange(len(lab)), subset(lab, fit_n, 0))
    idx = rest[subset(lab[rest], n, seed)]
    assert len(idx) == n, f"only {len(idx)} held-out images"
    return tuple(int(i) for i in idx)


def real_u8(rows, res, workers=8):
    """uint8 [B, res, res, 3]: the val JPEGs of these rows through ADM's centre crop at `res`."""
    from concurrent.futures import ThreadPoolExecutor
    paths = [str(val_root() / val_listing()[i]) for i in rows]
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(functools.partial(_crop, res=res), paths)))


def class_of(rows, res=64):
    import torch
    return torch.from_numpy(labels(res)[list(rows)])


# --- building the arrays ----------------------------------------------------------------------------------
def listing(src):
    paths = sorted(str(p) for p in Path(src).rglob("*") if p.suffix.lower() in _EXTS)
    syn = sorted({Path(p).relative_to(src).parts[0] for p in paths})
    index = {s: i for i, s in enumerate(syn)}
    lab = np.array([index[Path(p).relative_to(src).parts[0]] for p in paths], dtype=np.int16)
    assert len(paths) == N_VAL and len(syn) == CLASSES, (len(paths), len(syn))
    return paths, lab


def _crop(path, res):
    """ADM's centre crop, through EDM2's own dataset_tool so the pixels are bit-identical to its datasets."""
    import PIL.Image
    from ..models import _edm
    _edm.on_path(_edm.src("edm2"))
    import dataset_tool
    return dataset_tool.make_transform("center-crop-dhariwal", res, res)(np.array(PIL.Image.open(path).convert("RGB")))


def prep(res, src, workers=32, batch=64):
    import torch
    out_dir = root()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, lab = listing(src)
    shape = (len(paths), res, res, 3) if res == 64 else (len(paths), 8, res // 8, res // 8)
    name = "img64_val_u8.npy" if res == 64 else f"img{res}_val_vae.npy"
    tmp = out_dir / f"{name}.tmp.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8 if res == 64 else np.float32, shape=shape)
    if res == 512:
        from ..models import _edm
        _edm.edm_setup()
        from training.encoders import StabilityVAEEncoder
        enc = StabilityVAEEncoder(vae_name=_edm.hub.path(_edm.SD_VAE), batch_size=batch)
        encode = enc.encode_pixels                         # uint8 -> [0, 1] -> VAE, EDM2's convention
    elif res == 256:
        from diffusers import AutoencoderKL
        from ..models import _edm
        vae = AutoencoderKL.from_pretrained(_edm.hub.path(_edm.SD_VAE)).to("cuda").eval()

        def encode(x):
            d = vae.encode(x.float() / 127.5 - 1).latent_dist
            return torch.cat([d.mean, d.std], 1)
    buf, lo = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for i, arr in enumerate(pool.map(functools.partial(_crop, res=res), paths, chunksize=64)):
            if res == 64:
                out[i] = arr
                continue
            buf.append(arr)
            if len(buf) == batch or i == len(paths) - 1:
                x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).to("cuda")
                out[lo:lo + len(buf)] = encode(x).cpu().numpy()
                lo, buf = lo + len(buf), []
    out.flush()
    del out
    os.replace(tmp, out_dir / name)
    np.save(out_dir / f"img{res}_val_labels.npy", lab)
    print(f"wrote {out_dir / name}: {shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="build the ImageNet validation arrays")
    ap.add_argument("cmd", choices=["prep"])
    ap.add_argument("res", type=int, choices=[64, 256, 512])
    ap.add_argument("--src", required=True, help="the validation images, one folder per synset")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    prep(a.res, a.src, a.workers)
