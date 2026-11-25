"""CIFAR-10 for EDM's class-conditional and unconditional VP models (Fig 2's CIFAR-10 class-CFG pair).

    python -m vmg.data.cifar10 fetch       # the original batches -> $VMG_CIFAR10/cifar10_{train,test}_*.npy

The source is the original cifar-10-python.tar.gz, the archive EDM's dataset_tool.py reads, so images and
labels are in their original order. Calibration uses the 10,000 test images in that order; the train split is
the FID reference of EDM's protocol.
"""
import functools
import os
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .. import runs

NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"


def root():
    return Path(os.environ.get("VMG_CIFAR10", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "cifar10"))


@functools.lru_cache(maxsize=None)
def _arrays(split):
    return (np.load(root() / f"cifar10_{split}_u8.npy", mmap_mode="r"),
            np.load(root() / f"cifar10_{split}_labels.npy").astype(np.int64))


def n(split="test"):
    return len(_arrays(split)[1])


def pixels(rows, split="test"):
    """[B, 3, 32, 32] float32 in [-1, 1]."""
    import torch
    u8 = np.ascontiguousarray(_arrays(split)[0][list(rows)])
    return torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 127.5 - 1


def class_of(rows, split="test"):
    import torch
    return torch.from_numpy(_arrays(split)[1][list(rows)])


def fetch():
    import hashlib
    out = root()
    out.mkdir(parents=True, exist_ok=True)
    tgz = out / "cifar-10-python.tar.gz"
    if not tgz.exists():
        tmp = f"{tgz}.{os.getpid()}.tmp"
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, tgz)
    assert hashlib.md5(tgz.read_bytes()).hexdigest() == MD5, f"{tgz}: md5 mismatch (a short write: delete and retry)"
    with tarfile.open(tgz) as tf:
        get = lambda name: pickle.load(tf.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
        assert [n.decode() for n in get("batches.meta")[b"label_names"]] == NAMES
        for split, names in (("train", [f"data_batch_{i}" for i in range(1, 6)]), ("test", ["test_batch"])):
            bs = [get(n) for n in names]
            im = np.concatenate([b[b"data"] for b in bs]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            lab = np.concatenate([b[b"labels"] for b in bs]).astype(np.int64)
            np.save(out / f"cifar10_{split}_u8.npy", np.ascontiguousarray(im))
            np.save(out / f"cifar10_{split}_labels.npy", lab)
            print(f"{split}: {im.shape}, per class {np.bincount(lab)}")


if __name__ == "__main__":
    assert sys.argv[1:] == ["fetch"], __doc__
    fetch()
