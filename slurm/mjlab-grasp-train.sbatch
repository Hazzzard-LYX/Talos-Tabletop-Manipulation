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
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE}/projects/Talos-Tabletop-Manipulation}"
PYTHON="${WORKSPACE}/envs/mjlab-cu128/bin/python"
TASK="${TASK:-Mjlab-Tabletop-Grasp-Cube-Talos-v0}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
SEED="${SEED:-42}"
SHORT_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
RUN_NAME="${RUN_NAME:-${SHORT_REVISION}-${NUM_ENVS}env}"
RUN_ROOT="${WORKSPACE}/logs/talos-tabletop-training/${SHORT_REVISION}/${SLURM_JOB_ID}"
RESUME_CHECKPOINT_PATH="${RESUME_CHECKPOINT_PATH:-}"
INITIAL_STAGE="${INITIAL_STAGE:-0}"

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
echo "Initial curriculum stage: ${INITIAL_STAGE}"
nvidia-smi

test -x "${PYTHON}"
cd "${PROJECT_ROOT}"
TRAIN_ARGS=(
  "${TASK}"
  --env.scene.num-envs "${NUM_ENVS}"
  --env.seed "${SEED}"
  --agent.seed "${SEED}"
  --agent.logger tensorboard
  --agent.max-iterations "${MAX_ITERATIONS}"
  --agent.save-interval "${SAVE_INTERVAL}"
  --agent.run-name "${RUN_NAME}"
  --log-root "${RUN_ROOT}"
)

if [[ "${INITIAL_STAGE}" != "0" ]]; then
  TRAIN_ARGS+=(
    --env.curriculum.task-stage.params.initial-stage "${INITIAL_STAGE}"
  )
fi

if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  test -f "${RESUME_CHECKPOINT_PATH}"
  case "${TASK}" in
    *Cube*) EXPERIMENT_NAME="talos_tabletop_grasp_cube_privileged_position" ;;
    *Sphere*) EXPERIMENT_NAME="talos_tabletop_grasp_sphere_privileged_position" ;;
    *) echo "Cannot infer experiment name for resume task: ${TASK}" >&2; exit 2 ;;
  esac
  RESUME_RUN="resume_source"
  RESUME_DIR="${RUN_ROOT}/${EXPERIMENT_NAME}/${RESUME_RUN}"
  mkdir -p "${RESUME_DIR}"
  cp "${RESUME_CHECKPOINT_PATH}" "${RESUME_DIR}/model_resume.pt"
  TRAIN_ARGS+=(
    --agent.resume True
    --agent.load-run "${RESUME_RUN}"
    --agent.load-checkpoint "model_resume.pt"
  )
  echo "Resume checkpoint: ${RESUME_CHECKPOINT_PATH}"
fi

"${PYTHON}" slurm/train_grasp.py "${TRAIN_ARGS[@]}"

echo "TALOS_TABLETOP_TRAINING_OK"
echo "End: $(date --iso-8601=seconds)"
