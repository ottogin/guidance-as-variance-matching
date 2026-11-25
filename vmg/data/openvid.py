"""OpenVid-1M clips for the text-to-video models (Wan2.1, CogVideoX), as in paper tab:impl-data.

    python -m vmg.data.openvid fetch calib       # 56 clips from each of parts 30-70 (280; the first 256 calibrate)
    python -m vmg.data.openvid fetch eval        # 170 clips from each of parts 80-140 (the first 1,024 evaluate)

Only parts >= 29 hold full-resolution YouTube clips; a clip qualifies at >= 6.5 s and >= 15 fps (enough for
Wan's 81 frames at 16 fps and CogVideoX's 49 at 8 fps). Members are read out of the remote zips with HTTP range
requests, so only the chosen clips are downloaded. Each set is shuffled once with a fixed seed.
"""
import csv
import io
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .. import runs

REVISION = "d8a63bd22989c80b5734ec2bb989f4e1b61a5807"     # the dataset's commit the paper's clips were chosen at
HF = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/{REVISION}"
PARTS = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
SETS = {"calib": (PARTS[:5], 56), "eval": (PARTS[5:12], 170)}
N_CALIB, N_EVAL = 256, 1024
MIN_SECONDS, MIN_FPS = 6.5, 15.0


def root(split):
    base = Path(os.environ.get("VMG_OPENVID", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "openvid"))
    return base / split


def meta(split):
    return json.load(open(root(split) / "meta.json"))


def rows(split):
    return list(range(N_CALIB if split == "calib" else N_EVAL))


def captions(split, ids):
    m = meta(split)
    return [m[i]["caption"] for i in ids]


def read_clip(split, i, fps, n, H, W):
    """[3, n, H, W] float in [-1, 1]: the frames nearest t0 + j / fps, cover-resized and centre-cropped."""
    import av
    import torch
    import torch.nn.functional as F
    c = av.open(str(root(split) / "clips" / meta(split)[i]["video"]))
    s = c.streams.video[0]
    frames, times = [], []
    for fr in c.decode(s):
        frames.append(fr.to_ndarray(format="rgb24"))
        times.append(float(fr.pts * s.time_base))
        if times[-1] > times[0] + (n - 1) / fps + 0.5:      # no later frame can be nearer to a wanted time
            break
    c.close()
    times = np.asarray(times)
    want = times[0] + np.arange(n) / fps
    assert want[-1] <= times[-1] + 0.5 / fps, f"clip {i}: {times[-1] - times[0]:.2f} s < {n / fps:.2f} s"
    idx = np.abs(times[None] - want[:, None]).argmin(1)
    x = torch.from_numpy(np.stack([frames[j] for j in idx])).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    sc = max(H / h, W / w)
    hh, ww = max(H, round(h * sc)), max(W, round(w * sc))
    x = F.interpolate(x, size=(hh, ww), mode="bilinear", antialias=True, align_corners=False)
    t, l = (hh - H) // 2, (ww - W) // 2
    return (x[..., t:t + H, l:l + W] / 127.5 - 1).clamp(-1, 1).permute(1, 0, 2, 3).contiguous()


def save_mp4(frames, path, fps):
    """frames: [F, H, W, 3] float in [0, 1]."""
    import imageio
    tmp = f"{path}.tmp.mp4"
    imageio.mimwrite(tmp, (np.clip(frames, 0, 1) * 255).round().astype(np.uint8), fps=fps, quality=9)
    os.replace(tmp, path)


class _HttpFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests."""

    def __init__(self, url):
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=120)
        self.size, self.url, self.pos = int(r.headers["Content-Range"].split("/")[1]), r.geturl(), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, wh=0):
        self.pos = off if wh == 0 else (self.pos + off if wh == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(5):
            try:
                d = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}),
                                           timeout=300).read()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def fetch(split):
    from huggingface_hub import hf_hub_download
    parts, per = SETS[split]
    out = root(split)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    table = hf_hub_download("nkp37/OpenVid-1M", "data/train/OpenVid-1M.csv", repo_type="dataset", revision=REVISION)
    ok_rows = {}
    with open(table, newline="") as f:
        for r in csv.DictReader(f):
            if float(r["seconds"]) >= MIN_SECONDS and float(r["fps"]) >= MIN_FPS:
                ok_rows[r["video"]] = r
    rng, chosen = random.Random(0), []
    for p in parts:
        z = zipfile.ZipFile(io.BufferedReader(_HttpFile(f"{HF}/OpenVid_part{p}.zip"), buffer_size=1 << 22))
        members = {i.filename.split("/")[-1]: i for i in z.infolist() if not i.is_dir()}
        for m in rng.sample(sorted(m for m in members if m in ok_rows), per):
            dst = out / "clips" / m
            if not (dst.exists() and dst.stat().st_size == members[m].file_size):
                with z.open(members[m]) as src, open(f"{dst}.tmp", "wb") as o:
                    o.write(src.read())
                os.replace(f"{dst}.tmp", dst)
            assert dst.stat().st_size == members[m].file_size, f"{m}: short write"
            r = ok_rows[m]
            chosen.append(dict(video=m, part=p, caption=r["caption"], fps=float(r["fps"]), seconds=float(r["seconds"]),
                               frame=int(r["frame"])))
        print(f"part {p}: took {per}", flush=True)
    random.Random(1).shuffle(chosen)
    json.dump(chosen, open(out / "meta.json", "w"), indent=0)
    print(f"wrote {len(chosen)} clips to {out}")


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "fetch" and sys.argv[2] in SETS, __doc__
    fetch(sys.argv[2])
