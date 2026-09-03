"""Build diverse whole-body IK anchors for grasp Stage 1--20.

The output is a candidate bank.  Each candidate contains a complete physical
episode-start state, but it is not accepted as a permanent anchor until
``validate_reverse_curriculum_anchors.py`` has replayed it with the standing
policy and confirmed the initial physical Stage in the composed MJLab scene.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import talos_tabletop.tasks  # noqa: F401  # Register before MJLab plugin discovery.
import torch
from scipy.optimize import least_squares
from talos_tabletop.assets import object_initial_position
from talos_tabletop.robots.talos.constants import get_grasping_spec
from talos_tabletop.tasks.tabletop.env_cfg import (
  STABLE_CONTACT_GRIPPER_POSITION,
  talos_tabletop_stable_contact_lift_env_cfg,
)
from talos_tabletop.tasks.tabletop.mdp.reverse_curriculum import (
  DEFAULT_GRASP_STATE_LIMITS,
  GRASP_DIFFICULTY_COMPONENT_NAMES,
  GraspStateFeatures,
  classify_grasp_state,
)

SCHEMA_VERSION = 1
RIGHT_GRIPPER_RANGE_RAD = 0.959931
IK_JOINT_NAMES = (
  "torso_1_joint",
  "torso_2_joint",
  *(f"arm_right_{index}_joint" for index in range(1, 8)),
)
FAMILIES = ("axial", "lateral", "wrist", "gripper")


def _parse_stages(value: str) -> tuple[int, ...]:
  stages = tuple(int(item.strip()) for item in value.split(",") if item.strip())
  if not stages or any(stage < 1 or stage > 20 for stage in stages):
    raise argparse.ArgumentTypeError("Stages must be a comma-separated subset of 1--20.")
  if len(set(stages)) != len(stages):
    raise argparse.ArgumentTypeError("Stages cannot contain duplicates.")
  return stages


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
  axis = axis / np.linalg.norm(axis)
  skew = np.array(
    (
      (0.0, -axis[2], axis[1]),
      (axis[2], 0.0, -axis[0]),
      (-axis[1], axis[0], 0.0),
    )
  )
  return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (
    skew @ skew
  )


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, matrix.reshape(-1))
  return quat


def _rotation_error_vector(target: np.ndarray, current: np.ndarray) -> np.ndarray:
  error = np.empty(3)
  mujoco.mju_subQuat(error, _quat_from_matrix(target), _quat_from_matrix(current))
  return error


def _resolve_default_qpos(
  model: mujoco.MjModel,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
  cfg = talos_tabletop_stable_contact_lift_env_cfg(play=True)
  patterns = cfg.scene.entities["robot"].init_state.joint_pos or {}
  qpos = model.qpos0.copy()
  joint_names = []
  joint_qpos_addresses = []
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    joint_names.append(name)
    address = model.jnt_qposadr[joint_id]
    joint_qpos_addresses.append(address)
    for pattern, value in patterns.items():
      if re.fullmatch(pattern, name):
        qpos[address] = value
  return qpos, tuple(joint_names), np.asarray(joint_qpos_addresses)


def _set_coupled_gripper(
  model: mujoco.MjModel,
  qpos: np.ndarray,
  side: str,
  main_position: float,
) -> None:
  positions = {
    f"gripper_{side}_joint": main_position,
    f"gripper_{side}_inner_double_joint": main_position,
    f"gripper_{side}_motor_single_joint": -main_position,
    f"gripper_{side}_inner_single_joint": -main_position,
    f"gripper_{side}_fingertip_1_joint": -main_position,
    f"gripper_{side}_fingertip_2_joint": -main_position,
    f"gripper_{side}_fingertip_3_joint": -main_position,
  }
  for name, position in positions.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos[model.jnt_qposadr[joint_id]] = position


def _joint_limit_risk(
  model: mujoco.MjModel,
  qpos: np.ndarray,
  joint_ids: np.ndarray,
) -> float:
  positions = qpos[model.jnt_qposadr[joint_ids]]
  limits = model.jnt_range[joint_ids]
  normalized_margin = np.minimum(
    positions - limits[:, 0], limits[:, 1] - positions
  ) / (limits[:, 1] - limits[:, 0])
  # No risk in the middle 80%; rise linearly across the outermost 10%.
  return float(np.clip((0.10 - normalized_margin.min()) / 0.10, 0.0, 1.0))


def _solve_ik(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  seed_qpos: np.ndarray,
  qpos_addresses: np.ndarray,
  joint_ids: np.ndarray,
  site_id: int,
  target_position: np.ndarray,
  target_rotation: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
  seed = seed_qpos[qpos_addresses].copy()
  lower = model.jnt_range[joint_ids, 0] + 1.0e-3
  upper = model.jnt_range[joint_ids, 1] - 1.0e-3

  def residual(positions: np.ndarray) -> np.ndarray:
    data.qpos[:] = seed_qpos
    data.qpos[qpos_addresses] = positions
    mujoco.mj_forward(model, data)
    position_error = data.site_xpos[site_id] - target_position
    rotation_error = _rotation_error_vector(
      target_rotation, data.site_xmat[site_id].reshape(3, 3)
    )
    posture_regularization = positions - seed
    return np.concatenate(
      (10.0 * position_error, rotation_error, 0.03 * posture_regularization)
    )

  result = least_squares(
    residual,
    np.clip(seed, lower, upper),
    bounds=(lower, upper),
    max_nfev=250,
    ftol=1.0e-9,
    xtol=1.0e-9,
    gtol=1.0e-9,
  )
  residual(result.x)
  position_error = float(np.linalg.norm(data.site_xpos[site_id] - target_position))
  rotation_error = float(
    np.linalg.norm(
      _rotation_error_vector(target_rotation, data.site_xmat[site_id].reshape(3, 3))
    )
  )
  if not result.success or position_error > 0.003 or rotation_error > math.radians(3.0):
    return None
  solved_qpos = seed_qpos.copy()
  solved_qpos[qpos_addresses] = result.x
  return solved_qpos, position_error, rotation_error


def _stage_interval(stage: int) -> tuple[float, float]:
  return (stage - 1) / 20.0, stage / 20.0


def _expected_contact_links(stage: int) -> int:
  if stage == 1:
    return 2
  if stage == 2:
    return 1
  return 0


def _candidate_features(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos: np.ndarray,
  site_id: int,
  reference_position: np.ndarray,
  reference_rotation: np.ndarray,
  approach_axis: np.ndarray,
  controlled_joint_ids: np.ndarray,
  contact_links: int,
) -> tuple[GraspStateFeatures, np.ndarray, np.ndarray]:
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  delta = data.site_xpos[site_id] - reference_position
  signed_axial = float(delta @ approach_axis)
  lateral_vector = delta - signed_axial * approach_axis
  rotation_vector = _rotation_error_vector(
    reference_rotation, data.site_xmat[site_id].reshape(3, 3)
  )
  gripper_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_right_joint"
  )
  gripper_position = qpos[model.jnt_qposadr[gripper_joint_id]]
  joint_risk = _joint_limit_risk(model, qpos, controlled_joint_ids)
  features = GraspStateFeatures(
    axial_position_error_m=torch.tensor(abs(signed_axial)),
    lateral_position_error_m=torch.tensor(np.linalg.norm(lateral_vector)),
    wrist_rotation_error_rad=torch.tensor(np.linalg.norm(rotation_vector)),
    gripper_open_fraction=torch.tensor(abs(gripper_position) / RIGHT_GRIPPER_RANGE_RAD),
    joint_limit_risk=torch.tensor(joint_risk),
    projected_gravity_xy=torch.tensor(0.0),
    base_linear_speed_mps=torch.tensor(0.0),
    base_angular_speed_rad_s=torch.tensor(0.0),
    hand_object_relative_speed_mps=torch.tensor(0.0),
    contact_link_count=torch.tensor(contact_links),
    verified_grasp=torch.tensor(False),
  )
  signed_task_error = np.concatenate(
    (
      np.asarray((signed_axial,)),
      lateral_vector,
      rotation_vector,
      np.asarray((gripper_position / RIGHT_GRIPPER_RANGE_RAD, joint_risk)),
    )
  )
  return features, signed_task_error, rotation_vector


def _make_target(
  rng: np.random.Generator,
  stage: int,
  family: str,
  reference_position: np.ndarray,
  reference_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
  low, high = _stage_interval(stage)
  margin = min(0.008, 0.2 * (high - low))
  primary = rng.uniform(low + margin, high - margin)
  secondary_scale = rng.uniform(0.0, 0.12) * primary
  axial = secondary_scale * DEFAULT_GRASP_STATE_LIMITS.axial_position_error_m
  lateral = secondary_scale * DEFAULT_GRASP_STATE_LIMITS.lateral_position_error_m
  wrist = secondary_scale * DEFAULT_GRASP_STATE_LIMITS.wrist_rotation_error_rad
  gripper = secondary_scale

  if family == "axial":
    axial = primary * DEFAULT_GRASP_STATE_LIMITS.axial_position_error_m
  elif family == "lateral":
    lateral = primary * DEFAULT_GRASP_STATE_LIMITS.lateral_position_error_m
  elif family == "wrist":
    wrist = primary * DEFAULT_GRASP_STATE_LIMITS.wrist_rotation_error_rad
  elif family == "gripper":
    gripper = primary
  else:
    raise ValueError(f"Unknown anchor family: {family}")

  axial *= rng.choice((-1.0, 1.0))
  lateral_angle = rng.uniform(-math.pi, math.pi)
  lateral_local = np.asarray(
    (lateral * math.cos(lateral_angle), lateral * math.sin(lateral_angle), 0.0)
  )
  approach_axis = reference_rotation[:, 2]
  target_position = (
    reference_position
    + axial * approach_axis
    + reference_rotation @ lateral_local
  )

  rotation_axis_local = rng.normal(size=3)
  rotation_axis_local /= np.linalg.norm(rotation_axis_local)
  rotation_axis_world = reference_rotation @ rotation_axis_local
  wrist *= rng.choice((-1.0, 1.0))
  target_rotation = _rotation_matrix(rotation_axis_world, wrist) @ reference_rotation
  return target_position, target_rotation, float(np.clip(gripper, 0.0, 1.0))


def _farthest_point_selection(
  candidates: list[dict],
  count: int,
) -> list[dict]:
  descriptors = np.stack([candidate["diversity_descriptor"] for candidate in candidates])
  selected = []
  for family in FAMILIES:
    indices = [
      index for index, candidate in enumerate(candidates) if candidate["family"] == family
    ]
    if indices and len(selected) < count:
      center_distances = np.linalg.norm(descriptors[indices], axis=1)
      selected.append(indices[int(np.argmax(center_distances))])

  while len(selected) < count:
    remaining = [index for index in range(len(candidates)) if index not in selected]
    distances = np.linalg.norm(
      descriptors[remaining, None, :] - descriptors[np.asarray(selected), :], axis=-1
    )
    selected.append(remaining[int(np.argmax(distances.min(axis=1)))])
  return [candidates[index] for index in selected]


def _git_revision() -> str:
  try:
    return subprocess.check_output(
      ("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def build_candidates(
  *,
  stages: tuple[int, ...],
  anchors_per_stage: int,
  pool_multiplier: int,
  max_attempts_per_stage: int,
  seed: int,
) -> dict:
  model = get_grasping_spec().compile()
  data = mujoco.MjData(model)
  reference_qpos, joint_names, joint_qpos_addresses = _resolve_default_qpos(model)
  _set_coupled_gripper(
    model, reference_qpos, "right", STABLE_CONTACT_GRIPPER_POSITION
  )
  data.qpos[:] = reference_qpos
  mujoco.mj_forward(model, data)

  site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_center"
  )
  reference_position = data.site_xpos[site_id].copy()
  reference_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
  approach_axis = reference_rotation[:, 2].copy()
  ik_joint_ids = np.asarray(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      for name in IK_JOINT_NAMES
    ]
  )
  ik_qpos_addresses = model.jnt_qposadr[ik_joint_ids]
  object_position = object_initial_position("cube")
  object_root_state = np.asarray((*object_position, 1.0, 0.0, 0.0, 0.0, *([0.0] * 6)))
  robot_root_state = np.asarray((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, *([0.0] * 6)))
  rng = np.random.default_rng(seed)
  selected_anchors = []

  for stage in stages:
    pool = []
    target_pool_size = anchors_per_stage * pool_multiplier
    for attempt in range(max_attempts_per_stage):
      family = FAMILIES[attempt % len(FAMILIES)]
      target_position, target_rotation, gripper_open = _make_target(
        rng, stage, family, reference_position, reference_rotation
      )
      candidate_seed = reference_qpos.copy()
      candidate_seed[ik_qpos_addresses] += rng.normal(
        0.0, 0.015, size=len(ik_qpos_addresses)
      )
      solution = _solve_ik(
        model,
        data,
        candidate_seed,
        ik_qpos_addresses,
        ik_joint_ids,
        site_id,
        target_position,
        target_rotation,
      )
      if solution is None:
        continue
      qpos, ik_position_error, ik_rotation_error = solution
      _set_coupled_gripper(
        model, qpos, "right", -gripper_open * RIGHT_GRIPPER_RANGE_RAD
      )
      features, signed_error, _ = _candidate_features(
        model,
        data,
        qpos,
        site_id,
        reference_position,
        reference_rotation,
        approach_axis,
        ik_joint_ids,
        _expected_contact_links(stage),
      )
      classification = classify_grasp_state(features)
      if int(classification.stage.item()) != stage:
        continue

      joint_position = qpos[joint_qpos_addresses]
      normalized_ik_joints = (
        qpos[ik_qpos_addresses] - model.jnt_range[ik_joint_ids, 0]
      ) / (model.jnt_range[ik_joint_ids, 1] - model.jnt_range[ik_joint_ids, 0])
      descriptor = np.concatenate(
        (
          signed_error,
          normalized_ik_joints,
          classification.component_difficulties.detach().cpu().numpy(),
        )
      )
      pool.append(
        {
          "anchor_id": f"stage-{stage:02d}-candidate-{attempt:04d}",
          "requested_stage": stage,
          "provisional_stage": stage,
          "family": family,
          "permanent_anchor": True,
          "source": "whole_body_ik",
          "robot_root_state": torch.tensor(robot_root_state, dtype=torch.float32),
          "robot_joint_position": torch.tensor(joint_position, dtype=torch.float32),
          "robot_joint_velocity": torch.zeros(len(joint_names)),
          "object_root_state": torch.tensor(object_root_state, dtype=torch.float32),
          "last_action": torch.zeros(29),
          "difficulty": float(classification.difficulty.item()),
          "primary_difficulty": float(classification.primary_difficulty.item()),
          "secondary_difficulty": float(
            classification.secondary_difficulty.item()
          ),
          "component_difficulties": classification.component_difficulties
          .detach()
          .cpu(),
          "dominant_component": GRASP_DIFFICULTY_COMPONENT_NAMES[
            int(classification.dominant_component.item())
          ],
          "expected_contact_link_count": _expected_contact_links(stage),
          "signed_task_error": torch.tensor(signed_error, dtype=torch.float32),
          "ik_position_residual_m": ik_position_error,
          "ik_rotation_residual_rad": ik_rotation_error,
          "diversity_descriptor": descriptor,
          "standing_validation": None,
        }
      )
      if len(pool) >= target_pool_size:
        break

    if len(pool) < anchors_per_stage:
      raise RuntimeError(
        f"Stage {stage} produced only {len(pool)} valid IK candidates after "
        f"{max_attempts_per_stage} attempts."
      )
    chosen = _farthest_point_selection(pool, anchors_per_stage)
    for index, anchor in enumerate(chosen):
      anchor["anchor_id"] = f"stage-{stage:02d}-anchor-{index:02d}"
      anchor["diversity_descriptor"] = torch.tensor(
        anchor["diversity_descriptor"], dtype=torch.float32
      )
    selected_anchors.extend(chosen)
    print(
      f"Stage {stage:02d}: selected {len(chosen)} from {len(pool)} IK candidates",
      flush=True,
    )

  return {
    "schema_version": SCHEMA_VERSION,
    "kind": "reverse_curriculum_ik_anchor_candidates",
    "git_revision": _git_revision(),
    "seed": seed,
    "joint_names": joint_names,
    "component_names": GRASP_DIFFICULTY_COMPONENT_NAMES,
    "reference_grasp_position_w": torch.tensor(reference_position),
    "reference_grasp_quat_w": torch.tensor(_quat_from_matrix(reference_rotation)),
    "anchors": selected_anchors,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--stages", type=_parse_stages, default=tuple(range(1, 21)))
  parser.add_argument("--anchors-per-stage", type=int, default=10)
  parser.add_argument("--pool-multiplier", type=int, default=4)
  parser.add_argument("--max-attempts-per-stage", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=20260903)
  args = parser.parse_args()
  if args.anchors_per_stage <= 0:
    raise ValueError("anchors-per-stage must be positive.")
  if args.pool_multiplier <= 0:
    raise ValueError("pool-multiplier must be positive.")
  if args.max_attempts_per_stage < args.anchors_per_stage:
    raise ValueError("max-attempts-per-stage is smaller than anchors-per-stage.")

  bank = build_candidates(
    stages=args.stages,
    anchors_per_stage=args.anchors_per_stage,
    pool_multiplier=args.pool_multiplier,
    max_attempts_per_stage=args.max_attempts_per_stage,
    seed=args.seed,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(bank, args.output)
  print(f"Wrote {len(bank['anchors'])} candidate anchors to {args.output}")


if __name__ == "__main__":
  main()
