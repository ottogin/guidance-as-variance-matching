#!/usr/bin/env python
"""Download the EDM2 / EDM / ImageNet-256 checkpoints a model needs, rebuilding NVIDIA's unreleased post-hoc EMAs.

    python reproduce/fetch_weights.py --model edm2-64-cfg-s-xsu-fid     # one adapter's two networks
    python reproduce/fetch_weights.py                                   # every checkpoint (python -m vmg.models._edm fetch)
    python reproduce/fetch_weights.py --model edm2-64-cfg-s-xsu-fid --check   # smoke: every file is on the CDN, no download

A run_all.py job ahead of the floats whose networks are not on NVIDIA's CDN as they are used (vmg/models/_edm
rebuild: the run's raw snapshots, then edm2's reconstruct_phema.py), so a GPU job never downloads 30-160 GB itself.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmg.models import _edm, get  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", help="an EDM pair of vmg.models (repeatable; default: every checkpoint)")
    ap.add_argument("--check", action="store_true",
                    help="only check that every file (a rebuild's raw snapshots included) is downloadable, and that "
                         "the edm2 clone has reconstruct_phema.py")
    a = ap.parse_args()
    rels = [c for k in a.model for c in (get(k).net_ckpt, get(k).guide_ckpt)] if a.model else _edm.every_checkpoint()
    if not a.check:
        for rel in rels:
            (_edm.rebuild if "/posthoc-rebuilt/" in rel else _edm.fetch)(rel)
        return
    need = []
    for rel in rels:
        if "/posthoc-rebuilt/" in rel:
            run, kimg, _ = Path(rel).stem.rsplit("-", 2)
            need += _edm.snapshots(run, int(kimg))
            assert (_edm.src("edm2") / "reconstruct_phema.py").exists(), "the edm2 clone has no reconstruct_phema.py"
        else:
            need.append(rel)
    bad = []
    for rel in need:
        req = urllib.request.Request(_edm.url(rel), method="HEAD", headers={"User-Agent": "vmg"})
        try:
            urllib.request.urlopen(req, timeout=60).close()
        except Exception as e:                           # noqa: BLE001  (any failure means not downloadable)
            bad.append(f"{rel}: {e}")
    print(f"{len(need) - len(bad)} of {len(need)} files downloadable")
    if bad:
        raise SystemExit("\n".join(bad))


if __name__ == "__main__":
    main()
