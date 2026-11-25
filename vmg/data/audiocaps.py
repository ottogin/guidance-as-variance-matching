"""AudioCaps for AudioLDM 2 (paper tab:impl-data), from the OpenSound/AudioCaps mirror on the Hugging Face hub.

    python -m vmg.data.audiocaps fetch        # validation + test parquet -> clip arrays + meta.json (~6 GB)

One row per caption, in the mirror's file order: validation (2,223 captions of 445 clips) calibrates, test
(4,411 captions of 884 clips) evaluates. Captions of one clip share its audio, so each split is stored once
per clip: <split>_wav.npy (int16, all clips concatenated at their native rates; the mirror mixes 48 and
24 kHz), <split>_off.npy (offsets) and <split>_sr.npy, plus meta.json with every row's caption and clip.
"""
import glob
import io
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import runs

REPO = "OpenSound/AudioCaps"
REVISION = "b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded"     # the mirror's commit behind the paper's row order
SPLITS = ("validation", "test")
SPLIT_OF = {"calib": "validation", "eval": "test"}


def root():
    return Path(os.environ.get("VMG_AUDIOCAPS", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "audiocaps"))


@lru_cache(maxsize=1)
def meta():
    return json.load(open(root() / "meta.json"))


def _rows(split):
    return meta()["splits"][SPLIT_OF.get(split, split)]["rows"]


def rows(split):
    return list(range(len(_rows(split))))


def captions(split, ids):
    r = _rows(split)
    return [r[i]["caption"] for i in ids]


def clips(split, ids):
    r = _rows(split)
    return [r[i]["clip"] for i in ids]


@lru_cache(maxsize=2)
def _store(split):
    s = SPLIT_OF.get(split, split)
    flat = np.load(root() / f"{s}_wav.npy", mmap_mode="r")
    return flat, np.load(root() / f"{s}_off.npy"), np.load(root() / f"{s}_sr.npy")


def wav(split, clip):
    """(int16 [n], sample rate) of one clip, at its native rate (first channel)."""
    flat, off, sr = _store(split)
    return np.asarray(flat[off[clip]:off[clip + 1]]), int(sr[clip])


def _decode(b):
    import soundfile as sf
    info = sf.info(io.BytesIO(b))
    assert info.subtype == "PCM_16", info.subtype
    x, sr = sf.read(io.BytesIO(b), dtype="int16", always_2d=True)
    return x[:, 0].copy(), sr


def _iter_parquet(split):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(root() / "hf" / "data" / f"{split}-*.parquet")))
    assert files, f"no {split} parquet under {root() / 'hf'}; run fetch"
    for f in files:
        t = pq.read_table(f)
        cols = {c: t.column(c).to_pylist() for c in ("audiocap_id", "youtube_id", "start_time", "caption")}
        aud = t.column("audio").to_pylist()
        for i in range(t.num_rows):
            yield cols["audiocap_id"][i], cols["youtube_id"][i], cols["start_time"][i], cols["caption"][i], aud[i]["bytes"]


def fetch():
    from huggingface_hub import HfApi, hf_hub_download
    out = root()
    (out / "hf").mkdir(parents=True, exist_ok=True)
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
             if f.startswith(tuple(f"data/{s}-" for s in SPLITS))]
    for f in files:
        hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=out / "hf")
    meta_out = dict(seed=0, splits={}, source=f"{REPO}@{REVISION} (parquet)")
    for split in SPLITS:
        rows_, clips_, key2clip, wavs, srs, mismatch = [], [], {}, [], [], 0
        for aid, yid, st, cap, b in _iter_parquet(split):
            x, sr = _decode(b)
            k = (yid, int(st))
            if k in key2clip and not (srs[key2clip[k]] == sr and np.array_equal(wavs[key2clip[k]], x)):
                mismatch += 1
                k = (yid, int(st), len(rows_))          # same id, different audio: its own clip
            if k not in key2clip:
                key2clip[k] = len(wavs)
                wavs.append(x)
                srs.append(sr)
                clips_.append(dict(youtube_id=yid, start_time=int(st), n=len(x), sr=int(sr)))
            rows_.append(dict(id=int(aid), clip=key2clip[k], caption=cap))
        lens = np.asarray([len(x) for x in wavs])
        np.save(out / f"{split}_wav.npy", np.concatenate(wavs))
        np.save(out / f"{split}_off.npy", np.r_[0, np.cumsum(lens)])
        np.save(out / f"{split}_sr.npy", np.asarray(srs))
        meta_out["splits"][split] = dict(rows=rows_, clips=clips_, n_rows=len(rows_), n_clips=len(wavs),
                                         dup_audio_mismatch=mismatch)
        print(f"{split}: {len(rows_)} captions of {len(wavs)} clips ({mismatch} same-id/different-audio)", flush=True)
    tmp = out / f"meta.json.{os.getpid()}.tmp"
    json.dump(meta_out, open(tmp, "w"))
    os.replace(tmp, out / "meta.json")


if __name__ == "__main__":
    assert sys.argv[1:] == ["fetch"], __doc__
    fetch()
