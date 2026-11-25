"""A calibration set of your own: a folder of images (compute_w.py --dataset folder:<dir>).

Layout. `<dir>/index.jsonl` lists the examples, one JSON object per line:

    {"file": "img/0001.jpg", "caption": "a dog on a beach"}     text-to-image (CFG with the empty prompt)
    {"file": "img/0002.jpg", "label": 207}                      class-conditional (EDM2 CFG / autoguidance)
    {"file": "img/0003.jpg"}                                    unconditional

Without an index, every image under <dir> is a row, in sorted path order; an image inside a subfolder whose name
is an integer takes that class label. Row r is line r of the index (or entry r of the listing). Every row
calibrates; a folder has no evaluation split. Images are EXIF-oriented, centre-cropped to the render's aspect and
resized with Lanczos, as the paper's COCO photos were.

The calibration examples should be draws from the data the model is meant to generate, with their conditions:
the rule matches the guided prediction's variance to theirs.
"""
import functools
import json
from pathlib import Path

import numpy as np

PREFIX = "folder:"
SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def is_folder(dataset):
    return isinstance(dataset, str) and dataset.startswith(PREFIX)


def root(dataset):
    assert is_folder(dataset), dataset
    d = Path(dataset[len(PREFIX):]).expanduser()
    assert d.is_dir(), f"{d} is not a directory"
    return d


@functools.lru_cache(maxsize=None)
def index(dataset):
    """The examples: a list of dicts with "file" (absolute) and optionally "caption" / "label"."""
    d = root(dataset)
    idx = d / "index.jsonl"
    if idx.exists():
        out = []
        for i, line in enumerate(idx.read_text().splitlines()):
            if line.strip():
                e = json.loads(line)
                assert "file" in e, f"{idx}:{i + 1}: no 'file'"
                e = dict(e, file=str((d / e["file"]).resolve()))
                out.append(e)
    else:
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in SUFFIXES)
        out = []
        for p in files:
            e = {"file": str(p)}
            if p.parent != d and p.parent.name.isdigit():
                e["label"] = int(p.parent.name)
            out.append(e)
    assert out, f"no images in {d}"
    return out


def n(dataset):
    return len(index(dataset))


def has(dataset, key):
    return all(key in e for e in index(dataset))


def rows(dataset, split="calib"):
    assert split == "calib", f"{dataset}: a folder dataset has calibration rows only"
    return list(range(n(dataset)))


def captions(dataset, ids):
    ix = index(dataset)
    assert has(dataset, "caption"), f"{dataset}: index.jsonl has no 'caption' for every row"
    return [ix[i]["caption"] for i in ids]


def labels(dataset, ids):
    ix = index(dataset)
    assert has(dataset, "label"), f"{dataset}: no 'label' for every row (index.jsonl or integer subfolders)"
    return [int(ix[i]["label"]) for i in ids]


def image(dataset, i, size):
    """Row i as uint8 [H, W, 3]; `size` is an int (square) or (width, height)."""
    from PIL import Image, ImageOps
    W, H = (size, size) if np.isscalar(size) else size
    im = ImageOps.exif_transpose(Image.open(index(dataset)[i]["file"])).convert("RGB")
    w, h = im.size
    if w * H > h * W:                                   # wider than the render: crop the width
        c = h * W // H
        im = im.crop(((w - c) // 2, 0, (w - c) // 2 + c, h))
    else:
        c = w * H // W
        im = im.crop((0, (h - c) // 2, w, (h - c) // 2 + c))
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def uint8(dataset, ids, size):
    """[B, 3, H, W] uint8 torch tensor."""
    import torch
    return torch.from_numpy(np.stack([image(dataset, i, size) for i in ids])).permute(0, 3, 1, 2).contiguous()


def images(dataset, ids, size):
    """[B, 3, H, W] float32 in [-1, 1], the format of vmg.data.coco.images."""
    return uint8(dataset, ids, size).float() / 127.5 - 1


def split(dataset):
    """The Split a model lists for this folder."""
    from ..models.base import Split
    return Split(calib=f"{n(dataset):,} images of {root(dataset)}", n_calib=n(dataset))
