#!/usr/bin/env bash
#SBATCH --job-name=talos-anchor-build
#SBATCH --output=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.out
#SBATCH --error=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=stud
#SBATCH --gres=gpu:1

set -euo pipefail

WORKSPACE="${HOME}/IAS_Workspace"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE}/projects/Talos-Tabletop-Manipulation}"
PYTHON="${WORKSPACE}/envs/mjlab-cu128/bin/python"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the migrated stable-standing actor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/datasets/talos-tabletop-anchor-bank}"
CANDIDATES_PER_STAGE="${CANDIDATES_PER_STAGE:-40}"
ANCHORS_PER_STAGE="${ANCHORS_PER_STAGE:-10}"
SEED="${SEED:-20260903}"

export TMPDIR="${SLURM_TMPDIR:-${WORKSPACE}/tmp/${SLURM_JOB_ID}}"
export WARP_CACHE_PATH="${WORKSPACE}/shared/mjlab-cache/warp"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TMPDIR}" "${OUTPUT_ROOT}" "${WARP_CACHE_PATH}"

CANDIDATE_BANK="${OUTPUT_ROOT}/ik-anchor-candidates-${SLURM_JOB_ID}.pt"
FINAL_BANK="${OUTPUT_ROOT}/ik-anchor-bank-${SLURM_JOB_ID}.pt"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Git revision: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "Standing checkpoint: ${CHECKPOINT}"
nvidia-smi

cd "${PROJECT_ROOT}"
"${PYTHON}" slurm/build_reverse_curriculum_anchors.py \
  --output "${CANDIDATE_BANK}" \
  --anchors-per-stage "${CANDIDATES_PER_STAGE}" \
  --pool-multiplier 2 \
  --seed "${SEED}"

"${PYTHON}" slurm/validate_reverse_curriculum_anchors.py \
  --input "${CANDIDATE_BANK}" \
  --output "${FINAL_BANK}" \
  --checkpoint "${CHECKPOINT}" \
  --anchors-per-stage "${ANCHORS_PER_STAGE}" \
  --hold-duration-s 5.0 \
  --recovery-duration-s 2.0 \
  --device cuda:0 \
  --seed "${SEED}"

echo "ANCHOR_BANK=${FINAL_BANK}"
echo "TALOS_ANCHOR_BUILD_OK"
