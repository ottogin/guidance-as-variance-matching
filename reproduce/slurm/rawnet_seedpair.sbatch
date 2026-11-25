#!/bin/bash
# "EMA 0" of the seed-pair runs A and B: the raw training weights at 32 Mi, from each run's training-state checkpoint,
# in the format of a network snapshot (float16), at $OUT/phema/{A,B}/phema-0033554-0.000.pkl:
#
#     sbatch [cpu options] --chdir=<repo> reproduce/slurm/rawnet_seedpair.sbatch
#
# Before writing, it checks that the EMA stored in the checkpoint equals the 0.100 snapshot bit for bit, so that the raw
# weights are from the same training step as the snapshots.
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
source reproduce/slurm/site.env
OUT=$VMG_RUNS/edm2-64-seedpair
cd "$VMG_EDM2"
"$PY" - "$OUT" <<'EOF'
import os, pickle, sys
import torch
sys.path.insert(0, os.getcwd())
out = sys.argv[1]
flat = lambda ps: torch.cat([p.detach().float().flatten() for p in ps])
for r in "AB":
    with open(f"{out}/train/{r}/network-snapshot-0033554-0.100.pkl", "rb") as f:
        data = pickle.load(f)
    ck = torch.load(f"{out}/train/{r}/training-state-0033554.pt", map_location="cpu", weights_only=False)
    names = [n for n, _ in data.ema.named_parameters()]
    i = [round(float(s), 3) for s in ck["ema"]["stds"]].index(0.100)
    stored = ck["ema"]["emas"][i]
    snap = dict(data.ema.named_parameters())
    assert all(torch.equal(stored[n].to(torch.float16), snap[n]) for n in names), f"{r}: checkpoint is not the snapshot's step"
    ema = flat(snap.values())
    data.ema.load_state_dict(ck["net"], strict=True)                 # copies into the float16 parameters
    raw = flat(dict(data.ema.named_parameters()).values())
    os.makedirs(f"{out}/phema/{r}", exist_ok=True)
    p = f"{out}/phema/{r}/phema-0033554-0.000.pkl"
    with open(p + ".tmp", "wb") as f:
        pickle.dump(data, f)
    os.replace(p + ".tmp", p)
    print(f"{r}: step {ck['state']['cur_nimg']} images; EMA check passed; |raw - EMA 0.100| / |EMA 0.100| = "
          f"{(raw - ema).norm() / ema.norm():.4f}; wrote {p}", flush=True)
EOF
