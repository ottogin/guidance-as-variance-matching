#!/usr/bin/env python
"""Write the stored SD 3.5 Medium sweep metrics as evaluate.py outputs, to check the sweep tables and figures.

    python reproduce/checks/legacy_metrics.py --runs /tmp/legacy-runs
    VMG_RUNS=/tmp/legacy-runs python reproduce/tables/tab_tuned_w.py --out /tmp/t

Also SD 3.5 Medium's published-schedule arms (tab:pub-sched), SD 1.5's 0.5-grid sweep of fig:tuned-w-metric with its
pub and vm-global arms, and the EDM2-S ImageNet-512 0.5-grid sweep (as w<w> arms
of sd15/coco and edm2-512/imagenet512). The earlier pipeline scored every SD 3.5 Medium arm of the paper's search (10,000 COCO evaluation photos, 40
Euler steps) into results/gsweep/metrics/metrics-sd-coco-sd35-medium-<arm>-full-10k-o5705-s40-f32-pr.json. This
turns them into runs/sd35m/coco/metrics/<arm>.json: constants as w<w>, the unguided arm as w1, and the
paper's two "Ours" arms (the tr Cov rule: the constant 4.14 and the schedule) as vm-global and
vm-schedule. Only this check reads the old results/ tree, which is not part of this repository: it needs
results/gsweep/metrics/, results/sched/sd/, results/wgrid/sd15/, results/wgrid/edm2clip/ and
results/pr-table-512-s-cfg-fid-wsweep.json.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "gsweep" / "metrics"
PAT = "metrics-sd-coco-sd35-medium-*-full-10k-o5705-s40-f32-pr.json"
KEYS = ["fid", "fd_dinov2", "clip", "n"] + [f"{s}_{d}" for d in ("inception", "dinov2")
                                            for s in ("precision", "recall", "density", "coverage")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    out = Path(a.runs) / "sd35m" / "coco" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(str(SRC / PAT))):
        t = re.search(r"medium-(.*)-full-10k", f).group(1)
        if t == "unguided":
            arm = "w1"
        elif t == "wt-trace":
            arm = "vm-schedule"
        elif t.startswith("const"):
            arm = "vm-global" if t == "const4.14" else f"w{float(t[5:]):g}"
        else:
            continue
        d = json.load(open(f))
        w = d.get("w")
        if w is None:
            w = 1.0 if arm == "w1" else float(t[5:])
        w = np.atleast_1d(np.asarray(w, float))
        rec = {k: d[k] for k in KEYS if k in d}
        rec.update(model="sd35m", dataset="coco", arm=arm, legacy=Path(f).name, w=w.tolist(), w_mean=float(w.mean()))
        json.dump(rec, open(out / f"{arm}.json", "w"), indent=0)
        if arm == "vm-global":                       # 4.14 was also a point of the search grid
            json.dump(dict(rec, arm="w4.14"), open(out / "w4.14.json", "w"), indent=0)
        n += 1
    print(f"wrote {n} arms under {out}")
    sd15(Path(a.runs))
    edm2s(Path(a.runs))
    sched(Path(a.runs))


def sched(root):
    """SD 3.5 Medium's published-schedule arms of tab:pub-sched (results/sched/sd): the released constant as pub,
    each schedule at the released weight as pub-<kind> and at our constant as gw-<kind>."""
    out = root / "sd35m" / "coco" / "metrics"
    names = {"const4.5": "pub", **{f"wt-{b}-{k}": f"{b}-{k}" for b in ("pub", "gw") for k in ("interval", "linear", "tvcfg")}}
    n = 0
    for f in glob.glob(str(REPO / "results" / "sched" / "sd" / "**" / f"metrics-sd-coco-sd35-medium-*-full-10k-o5705-s40-f32-pr.json"),
                       recursive=True):
        t = re.search(r"medium-(.*)-full-10k", f).group(1)
        if t not in names:
            continue
        d = json.load(open(f))
        w = np.atleast_1d(np.asarray(d.get("w") if d.get("w") is not None else 4.5, float))
        rec = {k: d[k] for k in KEYS if k in d}
        rec.update(model="sd35m", dataset="coco", arm=names[t], legacy=Path(f).name, w=w.tolist(),
                   w_mean=float(d.get("w_mean") or w.mean()))
        json.dump(rec, open(out / f"{names[t]}.json", "w"), indent=0)
        n += 1
    print(f"wrote {n} schedule arms under {out}")


def sd15(root):
    """The SD 1.5 0.5-grid sweep (50 PNDM steps): results/wgrid/sd15/. The released 7.5 is a point of the grid and
    is also written as pub; our constant 7.44 is not, and is written as vm-global only (fig:tuned-w-metric's lines)."""
    out = root / "sd15" / "coco" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(str(REPO / "results" / "wgrid" / "sd15" / "metrics-sd-coco-sd15-*-full-10k-o5705-s50-pr.json"))):
        d = json.load(open(f))
        wn = d["wname"]
        w = 1.0 if wn == "unguided" else float(wn[5:]) if wn.startswith("const") else None
        if w is None or d.get("n") != 10000:
            continue
        rec = {k: d[k] for k in KEYS if k in d}
        rec.update(model="sd15", dataset="coco", legacy=Path(f).name, w_mean=w)
        arms = ([f"w{w:g}"] if abs(w * 2 - round(w * 2)) < 1e-9 else []) + {"const7.5": ["pub"], "const7.44": ["vm-global"]}.get(wn, [])
        for arm in arms:
            json.dump(dict(rec, arm=arm), open(out / f"{arm}.json", "w"), indent=0)
        n += bool(arms)
    print(f"wrote {n} arms under {out}")


def edm2s(root):
    """The EDM2-S ImageNet-512 0.5-grid sweep (NVIDIA's CFG preset for FID), with its CLIP scores."""
    out = root / "edm2-512" / "imagenet512" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for r in json.load(open(REPO / "results" / "pr-table-512-s-cfg-fid-wsweep.json"))["rows"]:
        w = float(r["mean_w"])
        if r.get("n") == 10000 and r["schedule"].startswith("const:") and r["schedule"] != "const:1.40" \
                and abs(w * 2 - round(w * 2)) < 1e-9:
            rows[w] = {k: r[k] for k in KEYS if k in r}
    for f in glob.glob(str(REPO / "results" / "wgrid" / "edm2clip" / "metrics-512-s-cfg-fid-const*-10k-clip.json")):
        m = json.load(open(f))
        w = float(re.search(r"-const([\d.]+)-10k-clip$", m["tag"]).group(1))
        if m.get("n") == 10000 and "clip" in m and abs(w * 2 - round(w * 2)) < 1e-9:
            rows.setdefault(w, {})["clip"] = m["clip"]
    for w, rec in rows.items():
        rec.update(model="edm2-512", dataset="imagenet512", arm=f"w{w:g}", w_mean=w)
        json.dump(rec, open(out / f"w{w:g}.json", "w"), indent=0)
    print(f"wrote {len(rows)} arms under {out}")


if __name__ == "__main__":
    main()
