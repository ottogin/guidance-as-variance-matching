"""Song Describer for Stable Audio Open (paper tab:impl-data), from Zenodo record 10072001 (CC-BY-SA).

    python -m vmg.data.songdescriber fetch       # song_describer.csv + audio.zip (3.3 GB, md5-checked) -> meta.json

Rows (meta.json):
    heldout  calibration: (caption, clip) on tracks with NO caption in the no-singing eval subset; a caption gives
             a clip at offset 0 s and one at 47 s, each kept only if the file holds all 47 s (356 captions of 260
             tracks -> 708 clips), sorted by (track, caption, offset)
    eval     the 586 no-singing captions (stable-audio-metrics' load/song_describer-nosinging.csv) to generate from
Audio is read in place from audio.zip, so nothing is extracted.
"""
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import runs

REC = "https://zenodo.org/api/records/10072001/files/{}/content"
FILES = {"song_describer.csv": "e90e9459c22bfbe69f5462dc1434d573", "audio.zip": "2126b8facfe9468cf806c6154e09bbe5"}
SIZES = {"audio.zip": 3318019617}
NOSINGING = ("https://raw.githubusercontent.com/Stability-AI/stable-audio-metrics/main/load/"
             "song_describer-nosinging.csv")
SR, SEC = 44100, 47
WIN = 2097152                              # the model's sample_size (47.55 s)
OFFSETS = (0, SEC)
SPLIT_OF = {"calib": "heldout", "eval": "eval"}


def root():
    return Path(os.environ.get("VMG_SONGDESCRIBER",
                               Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "songdescriber"))


@lru_cache(maxsize=1)
def meta():
    return json.load(open(root() / "meta.json"))


def rows(split):
    return list(range(len(meta()[SPLIT_OF[split]])))


def captions(split, ids):
    m = meta()[SPLIT_OF[split]]
    return [m[i]["caption"] for i in ids]


def eval_row(i):
    """dict(caption_id, track_id, caption) of evaluation row i."""
    return meta()["eval"][i]


def decode_mp3(data):
    """(float32 [C, n], sample rate) of mp3 bytes, at the native rate and channel layout."""
    import av
    with av.open(io.BytesIO(data)) as c:
        s = c.streams.audio[0]
        rs = av.AudioResampler(format="fltp")
        chunks = [o.to_ndarray() for fr in c.decode(s) for o in rs.resample(fr)]
        chunks += [o.to_ndarray() for o in rs.resample(None)]
        sr = s.codec_context.sample_rate
    return np.concatenate(chunks, axis=1).astype(np.float32), int(sr)


_ZIP, _TRACK = [None], {}


def track_audio(track_id):
    """[2, n] float32 torch at 44.1 kHz of a whole track (stable-audio-tools' prepare_audio: resample, mono ->
    stereo). The last track read is cached; calibration rows of one track are contiguous."""
    import torch
    from torchaudio import transforms as T
    key = str(track_id)
    if key not in _TRACK:
        if _ZIP[0] is None:
            _ZIP[0] = zipfile.ZipFile(root() / meta()["zip"])
        f = meta()["files"][key]
        a, sr = decode_mp3(_ZIP[0].read(f["member"]))
        assert sr == f["sr"] and a.shape == (f["channels"], f["frames"]), (key, sr, a.shape, f)
        x = torch.from_numpy(a)
        if sr != SR:
            x = T.Resample(sr, SR)(x)
        if x.shape[0] == 1:
            x = x.repeat(2, 1)
        _TRACK.clear()
        _TRACK[key] = x[:2].contiguous()
    return _TRACK[key]


def clip(split, i):
    """[2, WIN]: 47 s from the row's offset, clamped to [-1, 1] as the stable-audio-tools dataset does, zero-padded."""
    import torch
    r = meta()[SPLIT_OF[split]][i]
    a = track_audio(r["track_id"])
    lo = int(round(r["offset"] * SR))
    seg = a[:, lo:lo + SEC * SR]
    assert seg.shape[1] == SEC * SR, (r, a.shape)
    out = torch.zeros(2, WIN)
    out[:, :SEC * SR] = seg.clamp(-1, 1)
    return out


# ---------------------------------------------------------------------------------------------------- fetch

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _download(d, name):
    path = d / name
    if path.exists():
        return path
    part = Path(f"{path}.part")
    free = shutil.disk_usage(d).free / 2 ** 30
    assert free > SIZES.get(name, 0) / 2 ** 30 + 10, f"{free:.0f} GB free"
    for _ in range(3):
        subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "20", "-C", "-", "-sS", "-o", str(part),
                        REC.format(name)], check=False)
        if name not in SIZES or part.stat().st_size == SIZES[name]:
            break
    got = _md5(part)
    assert got == FILES[name], f"{name}: md5 {got} != {FILES[name]}"
    os.replace(part, path)
    return path


def fetch():
    d = root()
    d.mkdir(parents=True, exist_ok=True)
    allc = list(csv.DictReader(open(_download(d, "song_describer.csv"))))
    zpath = _download(d, "audio.zip")
    nos_path = d / "song_describer-nosinging.csv"
    if not nos_path.exists():                  # the pinned checkout's copy matches its references; else main's
        sam = Path(os.environ.get("VMG_SAO_METRICS", "/nonexistent")) / "load" / "song_describer-nosinging.csv"
        if sam.exists():
            shutil.copyfile(sam, nos_path)
        else:
            urllib.request.urlretrieve(NOSINGING, nos_path)
    nos = list(csv.DictReader(open(nos_path)))
    by_id = {r["caption_id"]: r for r in allc}
    assert all(r["caption_id"] in by_id and by_id[r["caption_id"]]["caption"] == r["caption"] for r in nos)
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".mp3")]
    key = lambda n: "/".join(n.split("/")[-2:]).split(".")[0]           # noqa: E731  '34/1004034[.2min].mp3'
    members = {key(n): n for n in names}
    assert len(members) == len(names), "two members for one track"
    files = {}
    with zipfile.ZipFile(zpath) as z:
        for r in allc:
            m = members[r["path"].split(".")[0]]
            if r["track_id"] not in files:
                a, sr = decode_mp3(z.read(m))
                files[r["track_id"]] = dict(member=m, sr=sr, channels=int(a.shape[0]), frames=int(a.shape[1]))
    eval_tracks = {r["track_id"] for r in nos}

    def clips(caps):
        out = []
        for r in sorted(caps, key=lambda r: (int(r["track_id"]), int(r["caption_id"]))):
            f = files[r["track_id"]]
            n44 = f["frames"] * SR // f["sr"]
            for off in OFFSETS:
                if n44 >= (off + SEC) * SR + SR // 100:          # 10 ms margin for the resampler's rounding
                    out.append(dict(caption_id=int(r["caption_id"]), track_id=int(r["track_id"]),
                                    caption=r["caption"], offset=off))
        return out

    heldout = clips([r for r in allc if r["track_id"] not in eval_tracks])
    ev = [dict(caption_id=int(r["caption_id"]), track_id=int(r["track_id"]), caption=r["caption"]) for r in nos]
    out = dict(zip=zpath.name, sec=SEC, sr=SR, offsets=list(OFFSETS), files=files, heldout=heldout,
               evalset=clips([by_id[r["caption_id"]] for r in nos]), eval=ev)
    tmp = d / f"meta.json.{os.getpid()}.tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, d / "meta.json")
    print(f"heldout {len(heldout)} clips, eval {len(ev)} captions -> {d / 'meta.json'}")


if __name__ == "__main__":
    assert sys.argv[1:] == ["fetch"], __doc__
    fetch()
