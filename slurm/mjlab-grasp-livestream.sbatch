#!/usr/bin/env bash
#SBATCH --job-name=talos-table-live
#SBATCH --output=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.out
#SBATCH --error=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=stud

set -euo pipefail

WORKSPACE="${HOME}/IAS_Workspace"
PROJECT_ROOT="${WORKSPACE}/projects/Talos-Tabletop-Manipulation"
PYTHON="${WORKSPACE}/envs/mjlab-cu128/bin/python"
TRAIN_JOB_ID="${TRAIN_JOB_ID:?Submit with TRAIN_JOB_ID=<job-id>}"
TRAIN_REVISION="${TRAIN_REVISION:?Submit with TRAIN_REVISION=<12-char-revision>}"
TASK="${TASK:-Mjlab-Tabletop-Grasp-Cube-Talos-v0}"
CHECKPOINT_ITERATION="${CHECKPOINT_ITERATION:-25}"
NUM_ENVS="${NUM_ENVS:-1}"
REMOTE_PORT="${REMOTE_PORT:-18080}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"
TRAIN_ROOT="${WORKSPACE}/logs/talos-tabletop-training/${TRAIN_REVISION}/${TRAIN_JOB_ID}"

export TMPDIR="${SLURM_TMPDIR:-${WORKSPACE}/tmp/${SLURM_JOB_ID}}"
export WARP_CACHE_PATH="${WORKSPACE}/shared/mjlab-cache/warp"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TMPDIR}" "${WARP_CACHE_PATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Task: ${TASK}"
echo "Waiting for model_${CHECKPOINT_ITERATION}.pt below ${TRAIN_ROOT}"

deadline="$((SECONDS + WAIT_SECONDS))"
checkpoint=""
while ((SECONDS < deadline)); do
  checkpoint="$(find "${TRAIN_ROOT}" -type f -name "model_${CHECKPOINT_ITERATION}.pt" -print -quit 2>/dev/null || true)"
  if [[ -n "${checkpoint}" ]]; then
    break
  fi
  sleep 10
done

if [[ -z "${checkpoint}" ]]; then
  echo "Timed out waiting for the training checkpoint." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" -u slurm/play_grasp_livestream.py \
  --checkpoint "${checkpoint}" \
  --task "${TASK}" \
  --num-envs "${NUM_ENVS}" \
  --port "${REMOTE_PORT}" \
  --device cpu
