#!/bin/bash
# One EDM2-XS ImageNet-64 training run of reproduce/experiments/seedpair_in64.py, with NVIDIA's stock recipe:
#
#     sbatch [slurm options] --gres=gpu:4 --chdir=<repo> reproduce/slurm/train_seedpair.sbatch [--smoke] NAME [BATCH_GPU]
#
# --smoke: 128 Ki images at batch 64, snapshots every 64 Ki, no gate (the plumbing, on prep --per-class's small zip).
# NAME is A or B (EDM2-XS) or S1 (EDM2-S), with seeds in seedpair_in64.TRAIN. First the data gate (labels and reference statistics match
# NVIDIA's; it runs once and is skipped afterwards), then train_edm2.py with the preset to 32 Mi. Rerunning
# the same command resumes from the latest training-state in the run directory (EDM2's own resume). Also runs
# outside Slurm (bash, as run_all.py's local launcher does), with site.env optional there.
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
if [ -f reproduce/slurm/site.env ]; then source reproduce/slurm/site.env; fi
PY=${PY:-python}
JOB=${SLURM_JOB_ID:-local-$$}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # EDM2 resumes from our own training-state-*.pt with torch.load (torch >= 2.6 refuses)
SMOKE=0
if [ "${1:-}" = "--smoke" ]; then SMOKE=1; shift; fi
NAME=$1
case $NAME in   # S: 2.25x the parameters of XS, so a smaller per-GPU batch (the total stays 2048) and fewer 5.6 GB checkpoints
  S*) PRESET=edm2-img64-s;  BGPU=${2:-128}; CKPT=4Mi;;
  *)  PRESET=edm2-img64-xs; BGPU=${2:-256}; CKPT=2Mi;;
esac
SEED=$("$PY" -c "import sys; sys.path.insert(0, '.'); from reproduce.experiments.seedpair_in64 import TRAIN; print(TRAIN['$NAME'])")
NGPU=$(nvidia-smi -L | wc -l)
[ "$NGPU" -lt 4 ] && CKPT=2Mi   # few GPUs (e.g. one): 4 Mi of S would be ~3 h between checkpoints
if [ "$SMOKE" = 1 ]; then
  LAST=0000131; ARGS=(--duration=128Ki --batch=64 --batch-gpu=$((64 / NGPU)) --status=16Ki --snapshot=64Ki --checkpoint=64Ki)
else
  LAST=0033554; ARGS=(--duration=32Mi --batch-gpu="$BGPU" --checkpoint="$CKPT")
fi
OUT=$VMG_RUNS/edm2-64-seedpair
echo "$(date '+%F %T') $(hostname) ${CUDA_VISIBLE_DEVICES:-}: run $NAME ($PRESET) seed $SEED on $NGPU GPUs"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader
mkdir -p "$OUT/train"
# The same run may be queued on two partitions: whichever starts first trains. A job
# exits at once if the run is finished, or if the job holding the run's lock is still alive.
if [ -f "$OUT/train/$NAME/network-snapshot-$LAST-0.050.pkl" ] && [ -f "$OUT/train/$NAME/network-snapshot-$LAST-0.100.pkl" ]; then
  echo "run $NAME already has its last snapshots ($LAST kimg)"; exit 0
fi
LOCK="$OUT/train/$NAME.lock"
# The lock holds "JOBID POWER", POWER = GPUs x per-GPU speed (measured on S: H200 2.5 s/kimg per GPU, L40S 8.2). A job
# with more POWER takes the run over from a live holder with less (it cancels the holder, losing at most one checkpoint
# interval of its work); otherwise the later job exits. The lock is rewritten before the cancel, so a job released by
# an afterany on the cancelled holder sees the new holder and exits.
case $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader) in
  *H200*) PW=10;; *H100*) PW=8;; *A100*) PW=5;; *L40S*) PW=3;; *) PW=2;;
esac
POWER=$((NGPU * PW))
if [ -f "$LOCK" ]; then
  read -r holder hpow < "$LOCK" || true
  hpow=${hpow:-$POWER}
  if [ "$holder" != "$JOB" ] && squeue -h -j "$holder" -t R 2>/dev/null | grep -q .; then
    if [ "$hpow" -ge "$POWER" ]; then echo "run $NAME is being trained by job $holder (power $hpow >= $POWER); exiting"; exit 0; fi
    echo "$JOB $POWER" > "$LOCK"
    echo "$(date '+%F %T') taking run $NAME over from job $holder (power $hpow < $POWER)"; scancel "$holder"
    for _ in $(seq 60); do squeue -h -j "$holder" 2>/dev/null | grep -q . || break; sleep 10; done
  fi
fi
echo "$JOB $POWER" > "$LOCK"
[ "$SMOKE" = 1 ] || "$PY" -u reproduce/experiments/seedpair_in64.py gate --ngpu "$NGPU"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader >> "$OUT/train/gpus-$NAME.txt"  # no pipe: head -1 SIGPIPEs nvidia-smi under pipefail
RUN=$OUT/train/$NAME
# A training state cut short (e.g. by a full disk) is smaller than the one before it; EDM2 would resume from it and crash.
mapfile -t ST < <(ls -1 "$RUN"/training-state-*.pt 2>/dev/null | sort)
if [ ${#ST[@]} -ge 2 ] && [ "$(stat -c %s "${ST[-1]}")" -lt "$(stat -c %s "${ST[-2]}")" ]; then
  echo "removing truncated ${ST[-1]}"; rm -f "${ST[-1]}"
fi
case $NAME in    # S states are 5.6 GB each: keep the newest two (the older one in case the newest is cut short)
  S*) ( set +eo pipefail; while sleep 300; do ls -1 "$RUN"/training-state-*.pt 2>/dev/null | sort | head -n -2 | xargs -r rm -f; done ) &;;
esac
cd "$VMG_EDM2"
"$PY" -u -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train_edm2.py \
    --outdir="$RUN" --data="$OUT/img64/img64.zip" --preset="$PRESET" --cond=True --seed="$SEED" "${ARGS[@]}"
