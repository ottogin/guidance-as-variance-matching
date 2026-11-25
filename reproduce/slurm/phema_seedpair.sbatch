#!/bin/bash
# Post-hoc EMA reconstructions of the seed-pair runs A and B at 32 Mi (reproduce/experiments/seedpair_in64.py),
# with EDM2's own reconstruct_phema.py over each run's saved snapshots (8 Mi apart, stds 0.050 and 0.100):
#
#     sbatch [cpu options] --chdir=<repo> reproduce/slurm/phema_seedpair.sbatch [--smoke] STDS
#
# --smoke: at the 131 kimg end of train_seedpair.sbatch --smoke's runs instead of 32 Mi.
# STDS is a comma list (e.g. 0.035,0.075,0.125,0.150,0.200). Writes $OUT/phema/{A,B}/phema-0033554-<std>.pkl.
# Then a leave-one-out accuracy check: rebuild 0.100 from the 0.050 snapshots only (and 0.050 from 0.100 only) and
# print the relative weight error against the raw EMA that was left out.
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
if [ -f reproduce/slurm/site.env ]; then source reproduce/slurm/site.env; fi   # optional outside Slurm
PY=${PY:-python}
KIMG=33554
if [ "${1:-}" = "--smoke" ]; then KIMG=131; shift; fi
P=$(printf %07d "$KIMG")
STDS=$1
OUT=$VMG_RUNS/edm2-64-seedpair
cd "$VMG_EDM2"
for R in A B; do
  "$PY" reconstruct_phema.py --indir="$OUT/train/$R" --outdir="$OUT/phema/$R" --outstd="$STDS" --outkimg="$KIMG" --skip
  "$PY" reconstruct_phema.py --indir="$OUT/train/$R" --instd=0.050 --outdir="$OUT/phema-check/$R" --outstd=0.100 --outkimg="$KIMG" --skip
  "$PY" reconstruct_phema.py --indir="$OUT/train/$R" --instd=0.100 --outdir="$OUT/phema-check/$R" --outstd=0.050 --outkimg="$KIMG" --skip
done
"$PY" - "$OUT" "$P" <<'EOF'
import pickle, sys, torch
out, P = sys.argv[1], sys.argv[2]
def params(p):
    with open(p, "rb") as f:
        return torch.cat([t.detach().float().flatten() for t in pickle.load(f)["ema"].parameters()])
for r in "AB":
    raw = {s: params(f"{out}/train/{r}/network-snapshot-{P}-{s}.pkl") for s in ("0.050", "0.100")}
    for s, o in (("0.100", "0.050"), ("0.050", "0.100")):
        rec = params(f"{out}/phema-check/{r}/phema-{P}-{s}.pkl")
        print(f"{r}: std {s} rebuilt from the {o} snapshots only: |rec - raw| / |raw - raw_{o}| = "
              f"{(rec - raw[s]).norm() / (raw[s] - raw[o]).norm():.3f}   |rec - raw| / |raw| = {(rec - raw[s]).norm() / raw[s].norm():.2e}")
EOF
