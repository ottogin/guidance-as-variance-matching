"""DOCCI (Onoe et al. 2024, CC BY 4.0) as Fig 2 uses it: 2,000 landscape test photos, each with a long description.

    python -m vmg.data.docci fetch         # descriptions + images into $VMG_DOCCI (~17 GB tarball)

The set is frozen the way the passes behind the paper froze it: the test split sorted by example id, permuted
with seed 0, keeping landscape 4:3 photos at least 1,152 px wide (EXIF-oriented) until 2,000 are taken. Every
model reaches its resolution by downsampling alone: the flow models get the whole frame at 1152 x 864, SD 1.x a
centre square at 512 x 512. Two caption datasets share the photos: `docci` is the full description and
`docci-short` its first sentence (at least six words, else the first two sentences).

Rows 0-1,999 calibrate the SD 1.x and SD 3 pairs, rows 0-1,023 the other flow models. DOCCI is never sampled in
the paper; "eval" is rows 1,024-1,999.
"""
import functools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from .. import runs

URLS = ("https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines",
        "https://storage.googleapis.com/docci/data/docci_images.tar.gz")
N, MIN_WIDTH = 2000, 1152
FRAME = (1152, 864)          # (width, height): the whole 2048 x 1536 frame scaled by 0.5625
SQUARE = 512                 # SD 1.x: the centre square of the short side


def root():
    return Path(os.environ.get("VMG_DOCCI", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "docci"))


def first_sentence(text):
    """The description's opening sentence (at least 6 words, else the first two)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = parts[0]
    if len(out.split()) < 6 and len(parts) > 1:
        out = out + " " + parts[1]
    return out


@functools.lru_cache(maxsize=2)
def select(seed=0, split="test"):
    """The frozen 2,000-photo set (cached under $VMG_RUNS/_data; choosing it EXIF-opens each candidate)."""
    cache = runs.root() / "_data" / f"docci-{split}-seed{seed}.json"
    if cache.exists():
        return json.load(open(cache))
    from PIL import Image, ImageOps
    rows = [json.loads(line) for line in open(root() / "docci_descriptions.jsonlines")]
    rows = sorted((r for r in rows if r["split"] == split), key=lambda r: r["example_id"])
    picked = []
    for i in np.random.default_rng(seed).permutation(len(rows)):
        r = rows[i]
        w, h = ImageOps.exif_transpose(Image.open(root() / "images" / r["image_file"])).size
        if w <= h or abs(w / h - 4 / 3) > 0.01 or w < MIN_WIDTH:
            continue
        picked.append(dict(example_id=r["example_id"], image_file=r["image_file"], full=r["description"],
                           short=first_sentence(r["description"])))
        if len(picked) == N:
            break
    assert len(picked) == N, f"only {len(picked)} eligible {split} photos"
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    json.dump(picked, open(tmp, "w"))
    os.replace(tmp, cache)
    return picked


def rows(split, n_calib=N):
    return list(range(n_calib)) if split == "calib" else list(range(1024, N))


def captions(ids, short=False):
    sel = select()
    return [sel[i]["short" if short else "full"] for i in ids]


def image(i, mode="frame"):
    """Photo `i` as uint8 [H, W, 3], EXIF-oriented: the whole frame at 1152 x 864 ("frame"), or the centre
    square of the short side at 512 x 512 ("square"); LANCZOS, always a downsample."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(root() / "images" / select()[i]["image_file"])).convert("RGB")
    if mode == "square":
        w, h = im.size
        c = min(w, h)
        im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c)).resize((SQUARE, SQUARE),
                                                                                               Image.LANCZOS)
    else:
        assert mode == "frame", mode
        im = im.resize(FRAME, Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def images(ids, mode="frame"):
    """[B, 3, H, W] float32 in [-1, 1]."""
    import torch
    x = torch.from_numpy(np.stack([image(i, mode) for i in ids])).permute(0, 3, 1, 2)
    return x.float() / 127.5 - 1


def fetch():
    import tarfile
    import urllib.request
    d = root()
    d.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        f = d / url.rsplit("/", 1)[1]
        if not f.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, f)
    print("unpacking the images")
    tarfile.open(d / "docci_images.tar.gz").extractall(d)
    # a short-written DOCCI file can still open by its header: decode the set fully
    bad = []
    for i, r in enumerate(select()):
        try:
            image(i)
        except Exception as e:  # noqa: BLE001
            bad.append((r["image_file"], type(e).__name__))
    assert not bad, f"{len(bad)} photos do not decode, re-extract them: {bad[:5]}"
    print(f"DOCCI in {d}: {N} photos selected, all decode")


if __name__ == "__main__":
    assert sys.argv[1:] == ["fetch"], __doc__
    fetch()
