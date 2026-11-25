#!/bin/bash
# The measurement passes of reproduce/experiments/sizegap_in64.py for one EMA, one GPU:
#
#     sbatch [slurm options] --gres=gpu:1 -c 6 --mem=24G --chdir=<repo> reproduce/slurm/sizegap.sbatch EMA [smoke]
#
# EMA is 0.050 or 0.100. With "smoke" the passes run on 20 images and a few levels under the tag "smoke" and
# the report is printed, as a check of the code before the full passes.
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
source reproduce/slurm/site.env
EMA=$1; MODE=${2:-full}
echo "$(date '+%F %T') $(hostname) job $SLURM_JOB_ID: sizegap EMA $EMA ($MODE)"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader
S=reproduce/experiments/sizegap_in64.py
if [ "$MODE" = smoke ]; then
  "$PY" -u $S gram --ema "$EMA" --tag smoke --n 20 --batch 10 --levels 0,8,16,24 < /dev/null
  "$PY" -u $S tcorr --ema "$EMA" --tag smoke --n 20 --batch 10 --levels 0,8,16,24 < /dev/null
  "$PY" -u $S report --emas "$EMA" --tag smoke < /dev/null
else
  "$PY" -u $S gram --ema "$EMA" < /dev/null
  "$PY" -u $S tcorr --ema "$EMA" < /dev/null
fi
echo "$(date '+%F %T') done"
