#!/usr/bin/env bash
#SBATCH --job-name=talos-table-grasp
#SBATCH --output=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.out
#SBATCH --error=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=stud
#SBATCH --gres=gpu:1

set -euo pipefail

WORKSPACE="${HOME}/IAS_Workspace"
PROJECT_ROOT="${WORKSPACE}/projects/Talos-Tabletop-Manipulation"
PYTHON="${WORKSPACE}/envs/mjlab-cu128/bin/python"
TASK="${TASK:-Mjlab-Tabletop-Grasp-Cube-Talos-v0}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
SEED="${SEED:-42}"
SHORT_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
RUN_NAME="${RUN_NAME:-${SHORT_REVISION}-${NUM_ENVS}env}"
RUN_ROOT="${WORKSPACE}/logs/talos-tabletop-training/${SHORT_REVISION}/${SLURM_JOB_ID}"

export TMPDIR="${SLURM_TMPDIR:-${WORKSPACE}/tmp/${SLURM_JOB_ID}}"
export WARP_CACHE_PATH="${WORKSPACE}/shared/mjlab-cache/warp"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TMPDIR}" "${RUN_ROOT}" "${WARP_CACHE_PATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "Task: ${TASK}"
echo "Environments: ${NUM_ENVS}"
echo "Iterations: ${MAX_ITERATIONS}"
echo "Checkpoint interval: ${SAVE_INTERVAL}"
nvidia-smi

test -x "${PYTHON}"
cd "${PROJECT_ROOT}"
"${PYTHON}" slurm/train_grasp.py "${TASK}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --env.seed "${SEED}" \
  --agent.seed "${SEED}" \
  --agent.logger tensorboard \
  --agent.max-iterations "${MAX_ITERATIONS}" \
  --agent.save-interval "${SAVE_INTERVAL}" \
  --agent.run-name "${RUN_NAME}" \
  --log-root "${RUN_ROOT}"

echo "TALOS_TABLETOP_TRAINING_OK"
echo "End: $(date --iso-8601=seconds)"
