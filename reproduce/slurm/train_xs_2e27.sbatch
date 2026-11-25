#!/bin/bash
# One EDM2-XS ImageNet-64 training run of T = 2^27 images (reproduce/experiments/nested_in64.py; also the
# label-dropout runs of fig:cfg-regimes):
#
#     sbatch [slurm options] --gres=gpu:8 --chdir=<repo> reproduce/slurm/train_xs_2e27.sbatch \
#         [--smoke] NAME SEED [LABEL_DROPOUT]
#
# NAME is the run folder $VMG_RUNS/edm2-64-xs-2e27/train/NAME (the paper's: A 0 and B 1; DA 0 0.1 and DB 1 0.1).
# NVIDIA's preset edm2-img64-xs, changed only in its length and save intervals: batch 2048 over the GPUs present
# (256 per GPU at most, accumulating on fewer than 8), EMA snapshots (stds 0.050 and 0.100) every 2^20 images,
# training states every 2^23. LABEL_DROPOUT (default 0) zeroes a sample's class label with that probability
# (classifier-free guidance); it needs reproduce/patches/edm2_label_dropout.patch, applied once to a copy of the
# pinned edm2 commit under $VMG_RUNS/edm2-64-xs-2e27/src/ (a run without it trains on $VMG_EDM2 itself, stock).
# The data are seedpair_in64.py prep's zip, checked by its gate first (once). Rerunning the command resumes from
# the newest training state (EDM2's own resume); a finished run exits at once.
# --smoke: 128 Ki images at batch 64 into train/smoke-NAME, snapshots every 64 Ki (the plumbing, one GPU, no gate).
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
if [ -f reproduce/slurm/site.env ]; then source reproduce/slurm/site.env; fi   # optional outside Slurm
PY=${PY:-python}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # EDM2 resumes its training states with torch.load (torch >= 2.6 refuses)
SMOKE=0
if [ "${1:-}" = "--smoke" ]; then SMOKE=1; shift; fi
NAME=$1; SEED=$2; LD=${3:-0}
ROOT=$VMG_RUNS/edm2-64-xs-2e27
DATA=$VMG_RUNS/edm2-64-seedpair/img64/img64.zip
NGPU=$(nvidia-smi -L | wc -l)
if [ "$SMOKE" = 1 ]; then
  NAME=smoke-$NAME; DUR=128Ki; SNAP=64Ki; CKPT=64Ki; STATUS=16Ki; BATCH=64; BGPU=$((64 / NGPU)); LAST=0000131
else
  DUR=128Mi; SNAP=1Mi; CKPT=8Mi; STATUS=128Ki; BATCH=2048; BGPU=$((2048 / NGPU)); LAST=0134217
fi
[ "$BGPU" -gt 256 ] && BGPU=256
RUN=$ROOT/train/$NAME
echo "$(date '+%F %T') $(hostname) ${CUDA_VISIBLE_DEVICES:-}: run $NAME seed $SEED label dropout $LD on $NGPU GPUs"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader
if [ -f "$RUN/network-snapshot-$LAST-0.050.pkl" ] && [ -f "$RUN/network-snapshot-$LAST-0.100.pkl" ]; then
  echo "run $NAME is finished"; exit 0
fi
# A resumed run must keep its seed and label dropout (EDM2 writes training_options.json when it creates the run).
if [ -f "$RUN/training_options.json" ]; then
  "$PY" - "$RUN/training_options.json" "$SEED" "$LD" <<'EOF'
import json, sys
o = json.load(open(sys.argv[1]))
have = (int(o["seed"]), float(o["loss_kwargs"].get("label_dropout", 0.0)))
want = (int(sys.argv[2]), float(sys.argv[3]))
sys.exit(0 if have == want else f"{sys.argv[1]}: seed, label dropout {have}, not {want}")
EOF
fi
SRC=$VMG_EDM2
EXTRA=()
if [ "$("$PY" -c "print(float('$LD') > 0)")" = True ]; then
  PATCH=$PWD/reproduce/patches/edm2_label_dropout.patch
  COMMIT=$("$PY" -c "from vmg.models._edm import REPOS; print(REPOS['edm2'][1])")
  SRC=$ROOT/src/edm2-label-dropout
  if [ ! -d "$SRC" ]; then      # built once; concurrent jobs each build their own and the first rename wins
    mkdir -p "$ROOT/src"
    TMP=$(mktemp -d "$ROOT/src/.build.XXXXXX")
    git -C "$VMG_EDM2" archive "$COMMIT" | tar -x -C "$TMP"
    patch -p1 -d "$TMP" < "$PATCH"
    mv -T "$TMP" "$SRC" 2>/dev/null || rm -rf "$TMP"
  fi
  grep -q "label_dropout" "$SRC/training/training_loop.py"
  EXTRA=(--label-dropout="$LD")
fi
[ "$SMOKE" = 1 ] || "$PY" -u reproduce/experiments/seedpair_in64.py gate --ngpu "$NGPU"
mkdir -p "$ROOT/train"
# A training state cut short (e.g. by a full disk) is smaller than the one before it; EDM2 would resume it and crash.
mapfile -t ST < <(ls -1 "$RUN"/training-state-*.pt 2>/dev/null | sort)
if [ ${#ST[@]} -ge 2 ] && [ "$(stat -c %s "${ST[-1]}")" -lt "$(stat -c %s "${ST[-2]}")" ]; then
  echo "removing truncated ${ST[-1]}"; rm -f "${ST[-1]}"
fi
cd "$SRC"
"$PY" -u -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train_edm2.py \
    --outdir="$RUN" --data="$DATA" --preset=edm2-img64-xs --cond=True --seed="$SEED" ${EXTRA[@]+"${EXTRA[@]}"} \
    --duration="$DUR" --batch="$BATCH" --batch-gpu="$BGPU" --status="$STATUS" --snapshot="$SNAP" --checkpoint="$CKPT"
