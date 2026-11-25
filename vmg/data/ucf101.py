"""UCF-101 for Latte-XL/2 and SVD img2vid (paper tab:impl-data).

    python -m vmg.data.ucf101 fetch            # UCF-101.zip from the HF mirror quchenyuan/UCF101-ZIP, unzipped
    python -m vmg.data.ucf101 prep [--workers 16]   # meta.json: the seeded row order, labels and frame counts

Without the mirror, the official UCF101.rar (crcv.ucf.edu/data/UCF101.php) unpacked into $VMG_UCF101 gives the
same UCF-101/<class>/<clip>.avi tree; prep reports whether meta.json matches the paper's.

All 13,320 clips (101 classes, 320x240 at 25 fps) are put in one seeded order, which fixes every role:
  Latte  rows 0-9,999 evaluate (generated row k takes clip k's class), rows 10,000-13,319 calibrate (3,320)
  SVD    the first 3,072 rows long enough for 14 frames at 7 fps: 0-2,047 evaluate (their first frame is the
         conditioning image), 2,048-3,071 calibrate (1,024). SVD row ids index that list.
Frames are cut from the decoded clip on the fly: Latte takes its own TemporalRandomCrop(16 x 3) + linspace,
SVD 14 frames at 7 fps cropped to 16:9 (rows 30:210), both from a start drawn per clip with a fixed seed. A
clip that is not 320x240 (4 of them) is bilinearly resized first. This is the preparation behind the paper's
numbers, with meta.json md5 a026600d.
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import runs

N, SEED = 13320, 0
LATTE_FRAMES, LATTE_INTERVAL, LATTE_EVAL = 16, 3, 10000
SVD_FRAMES, SVD_FPS, SRC_FPS, SVD_EVAL, SVD_CALIB = 14, 7, 25.0, 2048, 1024
HF_ZIP = ("quchenyuan/UCF101-ZIP", "UCF-101.zip")


def root():
    return Path(os.environ.get("VMG_UCF101", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "ucf101"))


@lru_cache(1)
def meta():
    return json.load(open(root() / "meta.json"))


def rows(model, split="calib"):
    """Row ids for model "latte" or "svd"; svd ids index meta()["svd_rows"]."""
    assert split in ("calib", "eval"), split
    if model == "latte":
        return list(range(LATTE_EVAL, N)) if split == "calib" else list(range(LATTE_EVAL))
    assert model == "svd", model
    return list(range(SVD_EVAL, SVD_EVAL + SVD_CALIB)) if split == "calib" else list(range(SVD_EVAL))


def labels(rows_):
    lab = meta()["labels"]
    return [int(lab[r]) for r in rows_]


def decode(path):
    """Every frame of one clip, uint8 [T, H, W, 3] (PyAV, as torchvision.io.read_video)."""
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def latte_indices(T, i):
    """Latte's TemporalRandomCrop(16 * 3) + linspace(begin, end - 1, 16), the start seeded per clip."""
    rng = np.random.default_rng([SEED, 1, i])
    span = LATTE_FRAMES * LATTE_INTERVAL
    begin = int(rng.integers(0, max(0, T - span - 1) + 1))
    end = min(begin + span, T)
    assert end - begin >= LATTE_FRAMES, (T, begin, end)
    return np.linspace(begin, end - 1, LATTE_FRAMES, dtype=int)


def svd_span():
    return int(round((SVD_FRAMES - 1) * SRC_FPS / SVD_FPS))


def svd_indices(T, i):
    """14 frames at 7 fps from a 25 fps clip, start seeded per clip; None if the clip is too short."""
    span = svd_span()
    if T <= span:
        return None
    rng = np.random.default_rng([SEED, 2, i])
    begin = int(rng.integers(0, T - span))
    return begin + np.round(np.arange(SVD_FRAMES) * SRC_FPS / SVD_FPS).astype(int)


def _fit_240x320(v):
    if v.shape[1:3] == (240, 320):
        return v, False
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(f).resize((320, 240), Image.BILINEAR)) for f in v]), True


def _clip(r):
    m = meta()
    v = decode(root() / "UCF-101" / m["files"][r])
    assert len(v) == m["frames"][r], f"row {r}: {len(v)} frames decoded, meta.json says {m['frames'][r]}"
    return v, int(m["video_index"][r])


def latte_clip(r):
    """uint8 [16, 240, 320, 3] for Latte row r."""
    v, i = _clip(r)
    return _fit_240x320(v[latte_indices(len(v), i)])[0]


def svd_clip(k):
    """uint8 [14, 180, 320, 3] for SVD row k (16:9 crop, before the resize to 1024x576)."""
    v, i = _clip(meta()["svd_rows"][k])
    return _fit_240x320(v[svd_indices(len(v), i)])[0][:, 30:210]


def ucf_list(base):
    """(classes, relative paths, labels): a label is the rank of its sorted class folder (Latte's find_classes),
    files sorted within a class."""
    classes = sorted(d for d in os.listdir(base) if (base / d).is_dir())
    files, labs = [], []
    for c, name in enumerate(classes):
        for f in sorted(os.listdir(base / name)):
            if f.endswith(".avi"):
                files.append(f"{name}/{f}")
                labs.append(c)
    assert len(classes) == 101 and len(files) == N, (len(classes), len(files))
    return classes, files, np.asarray(labs, dtype=np.int64)


def _frames(args):
    i, path = args
    v = decode(path)
    return len(v), v.shape[1:3] != (240, 320)


def prep(workers=16):
    """Write meta.json: the seeded order of every clip with its label and frame count, and the SVD rows."""
    from multiprocessing import Pool
    base = root() / "UCF-101"
    classes, files, labs = ucf_list(base)
    p = np.random.default_rng([SEED, 0]).permutation(N)
    with Pool(workers) as pool:
        got = pool.map(_frames, [(int(i), base / files[i]) for i in p], chunksize=16)
    T = [t for t, _ in got]
    svd_rows = [r for r in range(N) if svd_indices(T[r], int(p[r])) is not None][:SVD_EVAL + SVD_CALIB]
    m = dict(classes=classes, files=[files[i] for i in p], labels=labs[p].tolist(), video_index=p.tolist(), frames=T,
             svd_rows=svd_rows, svd_eval=SVD_EVAL, svd_mom=SVD_CALIB, latte_eval=LATTE_EVAL, seed=SEED,
             resized_rows=[r for r in range(N) if got[r][1]],
             latte=dict(frames=16, size=256, interval=3, steps=250, cfg=7.0, null=101, classes=101, scale=0.18215),
             svd=dict(frames=14, height=576, width=1024, steps=25, wmin=1.0, wmax=3.0, fps=7, motion_bucket_id=127,
                      noise_aug=0.02, src_fps=25.0),
             svd_span=svd_span())
    import hashlib
    tmp = root() / "meta.json.tmp"
    json.dump(m, open(tmp, "w"))
    os.replace(tmp, root() / "meta.json")
    md5 = hashlib.md5((root() / "meta.json").read_bytes()).hexdigest()
    print(f"wrote {root() / 'meta.json'}: {N} clips, {len(svd_rows)} SVD rows, {len(m['resized_rows'])} resized, "
          f"md5 {md5[:8]} ({'the paper' if md5.startswith('a026600d') else 'NOT the paper, a026600d'})")


def fetch():
    """Every clip of the HF mirror's UCF-101.zip that is missing locally or has the wrong size, read out of the
    remote zip with range requests; so it also completes or repairs a partial copy."""
    import io
    import zipfile
    from .openvid import _HttpFile
    url = f"https://huggingface.co/datasets/{HF_ZIP[0]}/resolve/main/{HF_ZIP[1]}"
    z = zipfile.ZipFile(io.BufferedReader(_HttpFile(url), buffer_size=1 << 22))
    got = 0
    members = [i for i in z.infolist() if i.filename.endswith(".avi")]
    assert len(members) == N, f"{len(members)} clips in the zip"
    for i in members:
        dst = root() / i.filename
        if dst.exists() and dst.stat().st_size == i.file_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with z.open(i) as src, open(f"{dst}.tmp", "wb") as o:
            o.write(src.read())
        os.replace(f"{dst}.tmp", dst)
        assert dst.stat().st_size == i.file_size, f"{dst}: short write"
        got += 1
    print(f"UCF-101: {N} clips in {root() / 'UCF-101'} ({got} fetched, the rest already complete)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch()
    elif cmd == "prep":
        prep(int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16)
    else:
        sys.exit(__doc__)
