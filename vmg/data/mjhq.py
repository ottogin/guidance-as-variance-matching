"""MJHQ-30K (Playground, 2023) as the paper uses it for FLUX.2 [klein]: 30,000 1024 x 1024 images in 10 categories.

    python -m vmg.data.mjhq fetch          # meta_data.json + images from the HF dataset playgroundai/MJHQ-30K

The image keys are sorted, then permuted with seed 0; prompts are used verbatim. Rows 0-1,023 calibrate and rows
5,705-15,704 evaluate (the same evaluation slice as COCO's).
"""
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

from .. import runs

N_CALIB, EVAL0, N_EVAL = 1024, 5705, 10000


def root():
    return Path(os.environ.get("VMG_MJHQ", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "mjhq"))


@functools.lru_cache(maxsize=2)
def select(seed=0):
    meta = json.load(open(root() / "meta_data.json"))
    keys = sorted(meta)
    return [dict(image_id=keys[i], file_name=f"{meta[keys[i]]['category']}/{keys[i]}.jpg",
                 caption=meta[keys[i]]["prompt"]) for i in np.random.default_rng(seed).permutation(len(keys))]


def rows(split):
    return list(range(N_CALIB)) if split == "calib" else list(range(EVAL0, EVAL0 + N_EVAL))


def captions(ids):
    sel = select()
    return [sel[i]["caption"] for i in ids]


def images(ids, size=1024):
    """[B, 3, size, size] float32 in [-1, 1] (the images are already 1024^2 squares)."""
    import torch
    from PIL import Image
    out = []
    for i in ids:
        im = Image.open(root() / select()[i]["file_name"]).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.LANCZOS)
        out.append(np.asarray(im, dtype=np.uint8))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 127.5 - 1


def fetch():
    import zipfile
    from huggingface_hub import hf_hub_download
    d = root()
    d.mkdir(parents=True, exist_ok=True)
    for f in ("meta_data.json", "mjhq30k_imgs.zip"):
        p = hf_hub_download("playgroundai/MJHQ-30K", f, repo_type="dataset", local_dir=d)
        if p.endswith(".zip"):
            zipfile.ZipFile(p).extractall(d)
    print(f"MJHQ-30K in {d}: {len(select())} images")


if __name__ == "__main__":
    assert sys.argv[1:] == ["fetch"], __doc__
    fetch()
