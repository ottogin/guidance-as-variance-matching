"""Bridge, as IRASim splits it, for Cosmos-Predict2.5 robot/action-cond (paper tab:impl-data).

    python -m vmg.data.bridge fetch        # 33 GB tarball by resumable range requests; keeps annotation/ + val/test videos

Calibration: every full 13-frame chunk (start 0, 12, 24, ...) of the 1,737 validation episodes, of which 1,024 are
drawn once with a fixed seed and used teacher-forced (the ground-truth chunk is the clean video). Evaluation: 100
test episodes drawn once with a fixed seed (the Cosmos-Predict2.5 paper samples 100 at random), rolled out
autoregressively. Row ids index those two lists.
"""
import glob
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import runs

CHUNK, N_CALIB, N_EVAL, SEED = 12, 1024, 100, 0
URL = "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz"
SIZE = 33108454502                     # bytes of the tarball


def root():
    """The folder holding annotation/{train,val,test} and videos/{val,test}."""
    return Path(os.environ.get("VMG_BRIDGE", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "bridge"))


def episodes(split):
    return sorted(glob.glob(str(root() / "annotation" / split / "*.json")))


@lru_cache(1)
def calib_chunks():
    """(episode json, start frame) of the 1,024 calibration chunks."""
    rows = []
    for p in episodes("val"):
        n = len(json.load(open(p))["state"])
        rows += [(p, i) for i in range(0, n - CHUNK, CHUNK)]
    idx = np.sort(np.random.default_rng(SEED + 1).choice(len(rows), min(N_CALIB, len(rows)), replace=False))
    return [rows[i] for i in idx]


@lru_cache(1)
def eval_episodes():
    ep = episodes("test")
    return [ep[i] for i in np.sort(np.random.default_rng(SEED).choice(len(ep), N_EVAL, replace=False))]


def rows(split="calib"):
    assert split in ("calib", "eval"), split
    return list(range(N_CALIB if split == "calib" else N_EVAL))


def fetch(segments=64):
    """The CDN stalls a single stream ~12.7 GB in and slows past that, so the tarball comes in `segments` resumable
    ranges with a stall timeout; then only annotation/ and videos/{val,test} are extracted (skipping latent_videos/,
    ~30k small files)."""
    import shutil
    import tarfile
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    dl = root().parent / "_bridge_dl"
    dl.mkdir(parents=True, exist_ok=True)
    seg = -(-SIZE // segments)

    def get(i):
        a, b = i * seg, min((i + 1) * seg, SIZE) - 1
        f = dl / f"part.{i:03d}"
        for _ in range(500):
            have = f.stat().st_size if f.exists() else 0
            if have >= b - a + 1:
                return
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={a + have}-{b}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(f, "ab") as o:
                    shutil.copyfileobj(r, o, 1 << 20)
            except Exception:
                time.sleep(5)
        raise RuntimeError(f"segment {i} did not finish")

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, range(segments)))
    parts = [dl / f"part.{i:03d}" for i in range(segments)]
    assert sum(p.stat().st_size for p in parts) == SIZE, "tarball size mismatch"

    class Cat:                         # the parts as one stream
        def __init__(self):
            self.fs = iter(parts)
            self.f = open(next(self.fs), "rb")

        def read(self, n=-1):
            while True:
                d = self.f.read(n)
                if d or n == 0:
                    return d
                self.f.close()
                nxt = next(self.fs, None)
                if nxt is None:
                    return b""
                self.f = open(nxt, "rb")

    want = ("bridge/annotation/", "bridge/videos/val/", "bridge/videos/test/")
    with tarfile.open(fileobj=Cat(), mode="r|gz") as t:
        for m in t:
            k = m.name.find("bridge/")
            if k < 0 or not m.name[k:].startswith(want) or not m.isfile():
                continue
            out = root() / m.name[k + len("bridge/"):]
            out.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(m) as src, open(f"{out}.tmp", "wb") as o:
                shutil.copyfileobj(src, o)
            os.replace(f"{out}.tmp", out)
            assert out.stat().st_size == m.size, f"{out}: short write"
    for s in ("val", "test"):
        print(f"{s}: {len(episodes(s))} episodes, {len(os.listdir(root() / 'videos' / s))} videos")
    print(f"the downloaded parts in {dl} can be deleted")


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        sys.exit(__doc__)
