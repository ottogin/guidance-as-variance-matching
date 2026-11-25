#!/bin/bash
# One generic job for every stage:
#
#     sbatch [slurm options] --chdir=<repo> reproduce/slurm/job.sbatch [--env NAME] <script.py> [args ...]
#
# In the arguments, the token {task} becomes $SLURM_ARRAY_TASK_ID, so an array of K moments shards is
#     sbatch --array=0-7 ... reproduce/slurm/job.sbatch compute_w.py --model sd35m --stage moments --shard {task} --shards 8
# Site settings (the python, VMG_RUNS, VMG_WEIGHTS, data roots, HF_HOME) come from reproduce/slurm/site.env,
# which is not committed; copy site.example.env. Configuration goes in arguments, never in --export.
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
source reproduce/slurm/site.env
if [ "${1:-}" = "--env" ]; then           # --env NAME: use $PY_NAME from site.env (e.g. a video or audio env)
  var="PY_$2"; PY="${!var}"; shift 2
fi
args=()
for x in "$@"; do args+=("${x//\{task\}/${SLURM_ARRAY_TASK_ID:-0}}"); done
echo "$(date '+%F %T') $(hostname) ${CUDA_VISIBLE_DEVICES:-no-gpu}: ${args[*]}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true
exec "$PY" -u "${args[@]}"
