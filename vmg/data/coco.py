"""MS-COCO val2014 as the paper uses it (App. "Data"): one fixed photo order, 480 x 480 centre crops.

    python -m vmg.data.coco fetch          # downloads val2014 images + annotations into $VMG_DATA/coco (~6.6 GB)

Photos are those whose short side is at least 480 px, so a centre crop reaches 480 x 480 without upsampling,
held in one fixed random order (seed 0) with the eligible photos first; the prompt of each is the longest of
its five captions (ties: the earliest annotation). Rows 0-5,704 calibrate, rows 5,705-15,704 evaluate.
"""
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

from .. import runs

URLS = ("http://images.cocodataset.org/zips/val2014.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip")
RENDER = MIN_SIDE = 480
N_CALIB, N_EVAL = 5705, 10000


def root():
    return Path(os.environ.get("VMG_COCO", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "coco"))


@functools.lru_cache(maxsize=4)
def select(seed=0, split="val2014"):
    """Every val2014 photo with captions, in the paper's fixed order (cached under $VMG_RUNS/_data)."""
    cache = runs.root() / "_data" / f"coco-order-{split}-seed{seed}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.load(open(cache))
    j = json.load(open(root() / "annotations" / f"captions_{split}.json"))
    caps = {}
    for a in sorted(j["annotations"], key=lambda a: a["id"]):
        caps.setdefault(a["image_id"], []).append(" ".join(a["caption"].split()))
    imgs = sorted((i for i in j["images"] if i["id"] in caps), key=lambda i: i["id"])
    order = np.random.default_rng(seed).permutation(len(imgs))
    rows = []
    for i in order:
        r, cs = imgs[i], caps[imgs[i]["id"]]
        rows.append(dict(image_id=r["id"], file_name=r["file_name"], width=r["width"], height=r["height"],
                         eligible=bool(min(r["width"], r["height"]) >= MIN_SIDE),
                         caption=max(cs, key=lambda c: (len(c.split()), -cs.index(c)))))
    rows.sort(key=lambda r: not r["eligible"])            # eligible first, the random order kept within
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    json.dump(rows, open(tmp, "w"))
    os.replace(tmp, cache)
    return rows


def rows(split):
    """Row ids: "calib" = 0..5,704, "eval" = 5,705..15,704 (all eligible)."""
    n = sum(r["eligible"] for r in select())
    assert n >= N_CALIB + N_EVAL, f"only {n} eligible photos"
    return list(range(N_CALIB)) if split == "calib" else list(range(N_CALIB, N_CALIB + N_EVAL))


def captions(ids):
    sel = select()
    return [sel[i]["caption"] for i in ids]


def image(i, size=RENDER):
    """Photo `i` as uint8 [size, size, 3]: EXIF-oriented, centre-cropped to a square, then resized only if
    the square is not already `size` (LANCZOS)."""
    from PIL import Image, ImageOps
    r = select()[i]
    im = ImageOps.exif_transpose(Image.open(root() / "val2014" / r["file_name"])).convert("RGB")
    w, h = im.size
    c = min(w, h)
    im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c))
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def images(ids, size=RENDER):
    """[B, 3, size, size] float32 in [-1, 1]."""
    import torch
    x = torch.from_numpy(np.stack([image(i, size) for i in ids])).permute(0, 3, 1, 2)
    return x.float() / 127.5 - 1


def fetch():
    import urllib.request
    import zipfile
    d = root()
    d.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        z = d / url.rsplit("/", 1)[1]
        if not z.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, z)
        print(f"unpacking {z.name}")
        zipfile.ZipFile(z).extractall(d)
    print(f"COCO val2014 in {d}: {len(select())} captioned photos")


if __name__ == "__main__":
    assert sys.argv[1:] == ["fetch"], __doc__
    fetch()
