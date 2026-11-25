#!/bin/bash
# A worker over a list of sampling arms of reproduce/experiments/seedpair_in64.py, one GPU:
#
#     sbatch [slurm options] --gres=gpu:1 -c 8 --mem=48G --chdir=<repo> reproduce/slurm/seedpair_worker.sbatch \
#         [--smoke] LIST [NEVAL]
#
# --smoke: A and B at train_seedpair.sbatch --smoke's 131 kimg snapshots, 8 seeds per arm.
# Each line of LIST is "ARM EMA ens_edm2.py-sample-args..." (e.g. "sw-AN-e100-w0.5 0.100 --d1 A --d0 N --w=0.5");
# EMA is a raw EMA (0.050, 0.100) or pSTD for a post-hoc reconstruction of A and B. Members: A, B (our XS), N, Nu (NVIDIA
# XS, cond and uncond), S1 (our EDM2-S), Sn (NVIDIA's EDM2-S); S1 and Sn exist at the raw EMAs only.
# For each arm not yet evaluated, the worker claims it (a lock in $OUT/locks holding its job id), samples all 10k
# seeds (PNGs already on disk are skipped) and runs the eval. Any number of workers, on any lane, can share one LIST:
# an arm whose lock holder is still running is skipped, and an arm whose holder died is taken over. Also runs
# outside Slurm (bash, as run_all.py's local launcher does), with site.env optional there.
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
if [ -f reproduce/slurm/site.env ]; then source reproduce/slurm/site.env; fi
PY=${PY:-python}
JOB=${SLURM_JOB_ID:-local-$$}
AB=0033554; NE=10000
if [ "${1:-}" = "--smoke" ]; then AB=0000131; NE=8; shift; fi
LIST=$1; NEVAL=${2:-$NE}
OUT=$VMG_RUNS/edm2-64-seedpair; RAW=${VMG_WEIGHTS:-$VMG_RUNS/_weights}/edm2/raw-snapshots   # vmg.models._edm's default
mkdir -p "$OUT/locks"
echo "$(date '+%F %T') $(hostname) job $JOB: worker over $LIST"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader
# An S1 list released before S1 finished training would fail: stop before claiming anything.
if grep -q -- "--d[01] S1" "$LIST" && [ ! -f "$OUT/train/S1/network-snapshot-0033554-0.100.pkl" ]; then
  echo "S1 has no 32 Mi snapshot yet; exiting"; exit 1
fi
while read -r ARM EMA ARGS; do
  [ -z "$ARM" ] && continue
  [ -f "$OUT/metrics/$ARM.json" ] && continue
  LK="$OUT/locks/$ARM"
  if ! mkdir "$LK" 2>/dev/null; then
    holder=$(cat "$LK/job" 2>/dev/null || echo none)
    if [ "$holder" != "$JOB" ] && squeue -h -j "$holder" -t R 2>/dev/null | grep -q .; then continue; fi
  fi
  echo "$JOB" > "$LK/job"
  case $EMA in     # pSTD: post-hoc EMA reconstruction (reproduce/slurm/phema_seedpair.sbatch); A and B only
    p*) E=${EMA#p}; PA=$OUT/phema/A/phema-$AB-$E.pkl; PB=$OUT/phema/B/phema-$AB-$E.pkl;;
    *)  E=$EMA; PA=$OUT/train/A/network-snapshot-$AB-$E.pkl; PB=$OUT/train/B/network-snapshot-$AB-$E.pkl;;
  esac
  EMA=$E
  CK="--ckpt A=$PA --ckpt B=$PB
      --ckpt N=$RAW/edm2-img64-xs/edm2-img64-xs-0033554-$EMA.pkl
      --ckpt Nu=$RAW/edm2-img64-xs-uncond/edm2-img64-xs-uncond-0033554-$EMA.pkl
      --ckpt S1=$OUT/train/S1/network-snapshot-0033554-$EMA.pkl
      --ckpt Sn=$RAW/edm2-img64-s/edm2-img64-s-0033554-$EMA.pkl --out $OUT --neval $NEVAL"
  echo "$(date '+%F %T') $ARM: sample"
  # shellcheck disable=SC2086
  "$PY" -u reproduce/experiments/ens_edm2.py sample --arm "$ARM" --shard 0 --shards 1 $CK $ARGS < /dev/null | tail -1
  echo "$(date '+%F %T') $ARM: eval"
  # shellcheck disable=SC2086
  "$PY" -u reproduce/experiments/ens_edm2.py eval --arm "$ARM" $CK < /dev/null
done < "$LIST"
echo "$(date '+%F %T') worker done"
