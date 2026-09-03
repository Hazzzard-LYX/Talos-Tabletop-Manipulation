"""Stage definitions for the grasping reverse curriculum.

The curriculum deliberately keeps two stage labels separate:

* ``reset_stage`` describes how a sampled initial state was generated.  It may
  include uncontrollable context such as object mass and friction.
* ``progress_stage`` describes only action-reducible grasp error.  It is used
  later for milestone rewards and rollout-state banking.

Stage zero is reserved for a verified grasp.  Stages 1--20 are ordered from
easiest to hardest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

PerturbationFocus = Literal[
  "gripper_opening",
  "axial_retreat",
  "lateral_position",
  "wrist_rotation",
  "arm_torso_joints",
  "base_state",
  "object_geometry",
  "object_dynamics",
]


@dataclass(frozen=True)
class ReverseCurriculumStageSpec:
  """Cumulative reset-distribution limits for one curriculum stage."""

  stage: int
  focus: PerturbationFocus
  level_in_focus: int
  max_gripper_opening_delta_rad: float = 0.0
  max_axial_retreat_m: float = 0.0
  max_lateral_position_error_m: float = 0.0
  max_wrist_rotation_error_rad: float = 0.0
  max_arm_torso_joint_error_rad: float = 0.0
  max_base_tilt_error_rad: float = 0.0
  max_base_linear_speed_mps: float = 0.0
  max_base_angular_speed_rad_s: float = 0.0
  max_object_xy_error_m: float = 0.0
  max_object_yaw_error_rad: float = 0.0
  object_size_scale_range: tuple[float, float] = (1.0, 1.0)
  object_mass_scale_range: tuple[float, float] = (1.0, 1.0)
  object_friction_scale_range: tuple[float, float] = (1.0, 1.0)


def _build_reverse_curriculum_stages() -> tuple[ReverseCurriculumStageSpec, ...]:
  cumulative: dict[str, float | tuple[float, float]] = {}
  schedule: tuple[
    tuple[int, PerturbationFocus, int, dict[str, float | tuple[float, float]]],
    ...,
  ] = (
    # 1--2: open the gripper relative to a successful Stage-0 grasp state.
    (1, "gripper_opening", 1, {"max_gripper_opening_delta_rad": 0.03}),
    (2, "gripper_opening", 2, {"max_gripper_opening_delta_rad": 0.08}),
    # 3--5: move the palm backward along the successful grasp approach axis.
    (3, "axial_retreat", 1, {"max_axial_retreat_m": 0.015}),
    (4, "axial_retreat", 2, {"max_axial_retreat_m": 0.040}),
    (5, "axial_retreat", 3, {"max_axial_retreat_m": 0.080}),
    # 6--7: introduce error perpendicular to that approach axis.
    (6, "lateral_position", 1, {"max_lateral_position_error_m": 0.008}),
    (7, "lateral_position", 2, {"max_lateral_position_error_m": 0.025}),
    # 8--10: rotate the wrist away from its successful object-relative pose.
    (8, "wrist_rotation", 1, {"max_wrist_rotation_error_rad": math.radians(5)}),
    (9, "wrist_rotation", 2, {"max_wrist_rotation_error_rad": math.radians(12)}),
    (10, "wrist_rotation", 3, {"max_wrist_rotation_error_rad": math.radians(25)}),
    # 11--12: disturb shoulder, elbow, and torso joints before solving/settling.
    (11, "arm_torso_joints", 1, {"max_arm_torso_joint_error_rad": 0.025}),
    (12, "arm_torso_joints", 2, {"max_arm_torso_joint_error_rad": 0.075}),
    # 13--14: disturb the floating-base attitude and initial velocity.
    (
      13,
      "base_state",
      1,
      {
        "max_base_tilt_error_rad": math.radians(2),
        "max_base_linear_speed_mps": 0.03,
        "max_base_angular_speed_rad_s": 0.08,
      },
    ),
    (
      14,
      "base_state",
      2,
      {
        "max_base_tilt_error_rad": math.radians(6),
        "max_base_linear_speed_mps": 0.10,
        "max_base_angular_speed_rad_s": 0.25,
      },
    ),
    # 15--17: randomize the cube pose and dimensions.
    (
      15,
      "object_geometry",
      1,
      {
        "max_object_xy_error_m": 0.015,
        "max_object_yaw_error_rad": math.radians(10),
        "object_size_scale_range": (0.95, 1.05),
      },
    ),
    (
      16,
      "object_geometry",
      2,
      {
        "max_object_xy_error_m": 0.040,
        "max_object_yaw_error_rad": math.radians(45),
        "object_size_scale_range": (0.90, 1.10),
      },
    ),
    (
      17,
      "object_geometry",
      3,
      {
        "max_object_xy_error_m": 0.080,
        "max_object_yaw_error_rad": math.pi,
        "object_size_scale_range": (0.80, 1.20),
      },
    ),
    # 18--20: widen object mass and contact-friction distributions.
    (
      18,
      "object_dynamics",
      1,
      {
        "object_mass_scale_range": (0.75, 1.25),
        "object_friction_scale_range": (0.85, 1.15),
      },
    ),
    (
      19,
      "object_dynamics",
      2,
      {
        "object_mass_scale_range": (0.50, 1.50),
        "object_friction_scale_range": (0.65, 1.35),
      },
    ),
    (
      20,
      "object_dynamics",
      3,
      {
        "object_mass_scale_range": (0.25, 2.00),
        "object_friction_scale_range": (0.50, 1.50),
      },
    ),
  )

  stages: list[ReverseCurriculumStageSpec] = []
  for stage, focus, level, updates in schedule:
    cumulative.update(updates)
    stages.append(
      ReverseCurriculumStageSpec(
        stage=stage,
        focus=focus,
        level_in_focus=level,
        **cumulative,
      )
    )
  return tuple(stages)


REVERSE_CURRICULUM_STAGES = _build_reverse_curriculum_stages()


def get_reverse_curriculum_stage(stage: int) -> ReverseCurriculumStageSpec:
  """Return the immutable reset specification for Stage 1--20."""
  if not 1 <= stage <= len(REVERSE_CURRICULUM_STAGES):
    raise ValueError("Reverse-curriculum stage must be in [1, 20].")
  return REVERSE_CURRICULUM_STAGES[stage - 1]


@dataclass(frozen=True)
class GraspProgressStageLimits:
  """Errors that saturate the controllable progress classifier at Stage 20."""

  gripper_opening_delta_rad: float = 0.10
  axial_position_error_m: float = 0.20
  lateral_position_error_m: float = 0.10
  wrist_rotation_error_rad: float = math.pi / 2.0


DEFAULT_GRASP_PROGRESS_LIMITS = GraspProgressStageLimits()


def classify_grasp_progress_stage(
  gripper_opening_delta_rad: torch.Tensor,
  axial_position_error_m: torch.Tensor,
  lateral_position_error_m: torch.Tensor,
  wrist_rotation_error_rad: torch.Tensor,
  grasp_formed: torch.Tensor,
  limits: GraspProgressStageLimits = DEFAULT_GRASP_PROGRESS_LIMITS,
) -> torch.Tensor:
  """Map every action-reducible grasp state to integer Stage 0--20.

  Stage 0 means that the caller's verified-grasp predicate is true.  Otherwise
  each normalized error is split into twenty equal intervals, and the worst
  component selects the stage.  This max-norm construction prevents a small
  error on one axis from compensating for a dangerous error on another.

  Mass, friction, object size, and reset-time disturbances are intentionally
  absent: the policy cannot reduce those values through action.  They belong
  to ``reset_stage`` and must not block within-episode progress rewards.
  """
  limit_values = (
    limits.gripper_opening_delta_rad,
    limits.axial_position_error_m,
    limits.lateral_position_error_m,
    limits.wrist_rotation_error_rad,
  )
  if any(value <= 0.0 for value in limit_values):
    raise ValueError("All grasp-progress limits must be positive.")

  errors = torch.broadcast_tensors(
    gripper_opening_delta_rad,
    axial_position_error_m,
    lateral_position_error_m,
    wrist_rotation_error_rad,
  )
  normalized = torch.stack(
    tuple(
      torch.abs(error) / limit
      for error, limit in zip(errors, limit_values, strict=True)
    ),
    dim=-1,
  )
  difficulty = normalized.amax(dim=-1)
  stage = torch.ceil(difficulty * 20.0).to(dtype=torch.long).clamp(1, 20)
  grasp_formed = torch.as_tensor(
    grasp_formed, device=stage.device, dtype=torch.bool
  ).expand_as(stage)
  return torch.where(grasp_formed, torch.zeros_like(stage), stage)
