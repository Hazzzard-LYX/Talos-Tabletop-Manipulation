"""Task-state classifier for the grasping reverse curriculum.

This first classifier partitions robot states, not perturbation sources or raw
joint coordinates. Every physically valid standing state maps to Stage 1--20,
while Stage 0 is reserved for a verified, dynamically stable grasp. The
initial analytic limits and equal-width boundaries are deliberately exposed so
rollout success data can calibrate them later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

GRASP_DIFFICULTY_COMPONENT_NAMES = (
  "axial_position",
  "lateral_position",
  "wrist_rotation",
  "gripper_opening",
  "joint_limit_risk",
  "base_tilt",
  "base_linear_speed",
  "base_angular_speed",
  "hand_object_relative_speed",
)

# Nineteen boundaries create twenty closed, ordered difficulty regions. They
# are an initialization only; later evaluation will replace them with values
# calibrated against empirical finite-horizon grasp success probability.
DEFAULT_GRASP_STAGE_BOUNDARIES = tuple(index / 20.0 for index in range(1, 20))

REVERSE_CURRICULUM_FIRST_STAGE = 1
REVERSE_CURRICULUM_FINAL_STAGE = 20
DEFAULT_REVERSE_CURRICULUM_PROMOTION_SUCCESS_RATE = 0.80
DEFAULT_REVERSE_CURRICULUM_EVALUATION_EPISODES = 4096
DEFAULT_REVERSE_CURRICULUM_SUCCESS_HOLD_DURATION_S = 5.0
DEFAULT_REVERSE_CURRICULUM_SUCCESS_LIFT_HEIGHT_M = 0.08


@dataclass(frozen=True)
class ReverseCurriculumStageDefinition:
  """Initialization mixture and promotion target for one curriculum stage.

  Difficulty indices increase from the verified-grasp seed at Stage 0 toward
  the hardest classified standing states at Stage 20. Promotion statistics
  use only episodes initialized from ``promotion_evaluation_stage``; easier
  replay and harder exploratory samples cannot bias the gate.
  """

  stage: int
  initialization_mixture: tuple[tuple[int, float], ...]
  success_hold_duration_s: float
  success_lift_height_m: float
  promotion_success_rate: float | None
  promotion_evaluation_stage: int | None
  evaluation_episodes: int | None

  def initialization_probability(self, difficulty_stage: int) -> float:
    """Return the reset probability assigned to one difficulty stage."""
    return next(
      (
        probability
        for candidate_stage, probability in self.initialization_mixture
        if candidate_stage == difficulty_stage
      ),
      0.0,
    )


def build_reverse_curriculum_stage_definitions(
  *,
  promotion_success_rate: float = (
    DEFAULT_REVERSE_CURRICULUM_PROMOTION_SUCCESS_RATE
  ),
  evaluation_episodes: int = DEFAULT_REVERSE_CURRICULUM_EVALUATION_EPISODES,
  success_hold_duration_s: float = (
    DEFAULT_REVERSE_CURRICULUM_SUCCESS_HOLD_DURATION_S
  ),
  success_lift_height_m: float = DEFAULT_REVERSE_CURRICULUM_SUCCESS_LIFT_HEIGHT_M,
) -> tuple[ReverseCurriculumStageDefinition, ...]:
  """Build Stage 1--20 without a hard switch between difficulty regions."""
  if not 0.0 < promotion_success_rate < 1.0:
    raise ValueError("Promotion success rate must lie strictly inside (0, 1).")
  if evaluation_episodes <= 0:
    raise ValueError("Evaluation episodes must be positive.")
  if success_hold_duration_s <= 0.0:
    raise ValueError("Success hold duration must be positive.")
  if success_lift_height_m <= 0.0:
    raise ValueError("Success lift height must be positive.")

  definitions = []
  for stage in range(
    REVERSE_CURRICULUM_FIRST_STAGE,
    REVERSE_CURRICULUM_FINAL_STAGE + 1,
  ):
    if stage < REVERSE_CURRICULUM_FINAL_STAGE:
      mixture = ((stage - 1, 0.30), (stage, 0.60), (stage + 1, 0.10))
      required_rate: float | None = promotion_success_rate
      promotion_evaluation_stage: int | None = stage
      required_episodes: int | None = evaluation_episodes
    else:
      mixture = ((stage - 1, 0.30), (stage, 0.70))
      required_rate = None
      promotion_evaluation_stage = None
      required_episodes = None

    definitions.append(
      ReverseCurriculumStageDefinition(
        stage=stage,
        initialization_mixture=mixture,
        success_hold_duration_s=success_hold_duration_s,
        success_lift_height_m=success_lift_height_m,
        promotion_success_rate=required_rate,
        promotion_evaluation_stage=promotion_evaluation_stage,
        evaluation_episodes=required_episodes,
      )
    )
  return tuple(definitions)


DEFAULT_REVERSE_CURRICULUM_STAGES = build_reverse_curriculum_stage_definitions()


def assess_reverse_curriculum_promotion(
  *,
  successful_current_stage_episodes: int,
  completed_current_stage_episodes: int,
  definition: ReverseCurriculumStageDefinition,
) -> tuple[float, bool]:
  """Return current-bin success rate and whether the next stage may open.

  The comparison is intentionally strict: exactly 80% does not satisfy a
  requirement that success probability be greater than 80%.
  """
  if completed_current_stage_episodes < 0:
    raise ValueError("Completed episode count cannot be negative.")
  if not 0 <= successful_current_stage_episodes <= completed_current_stage_episodes:
    raise ValueError("Successful episodes must be within the completed count.")

  success_rate = (
    successful_current_stage_episodes / completed_current_stage_episodes
    if completed_current_stage_episodes > 0
    else 0.0
  )
  if (
    definition.promotion_success_rate is None
    or definition.evaluation_episodes is None
  ):
    return success_rate, False
  ready = (
    completed_current_stage_episodes >= definition.evaluation_episodes
    and success_rate > definition.promotion_success_rate
  )
  return success_rate, ready


@dataclass(frozen=True)
class GraspStateFeatures:
  """Task-relevant coordinates for a batch of valid standing states.

  Position and rotation errors are measured against the desired object-relative
  grasp frame. ``gripper_open_fraction`` and ``joint_limit_risk`` are already
  normalized to [0, 1]. The contact count must count distinct hand links, not
  raw contact points. ``verified_grasp`` is a strict force-closure predicate;
  the classifier additionally checks motion before allowing Stage 0.
  """

  axial_position_error_m: torch.Tensor
  lateral_position_error_m: torch.Tensor
  wrist_rotation_error_rad: torch.Tensor
  gripper_open_fraction: torch.Tensor
  joint_limit_risk: torch.Tensor
  projected_gravity_xy: torch.Tensor
  base_linear_speed_mps: torch.Tensor
  base_angular_speed_rad_s: torch.Tensor
  hand_object_relative_speed_mps: torch.Tensor
  contact_link_count: torch.Tensor
  verified_grasp: torch.Tensor


def talos_gripper_open_fraction(
  joint_position: torch.Tensor,
  joint_range_rad: float = 0.959931,
) -> torch.Tensor:
  """Map TALOS gripper position to 0=closed and 1=open.

  The hardware/MJCF joint runs from approximately ``-0.96`` (closed) to
  ``0.0`` (open).  Using absolute joint position directly reverses the
  curriculum ordering.
  """
  return (1.0 - torch.abs(joint_position) / joint_range_rad).clamp(0.0, 1.0)


@dataclass(frozen=True)
class GraspStateDifficultyLimits:
  """Initial saturation and Stage-0 stability limits for the classifier."""

  axial_position_error_m: float = 0.30
  lateral_position_error_m: float = 0.20
  wrist_rotation_error_rad: float = math.pi
  base_tilt_projected_gravity_xy: float = 0.70
  base_linear_speed_mps: float = 0.50
  base_angular_speed_rad_s: float = 1.00
  hand_object_relative_speed_mps: float = 0.50
  secondary_difficulty_weight: float = 0.35
  stage_zero_max_projected_gravity_xy: float = 0.15
  stage_zero_max_base_linear_speed_mps: float = 0.10
  stage_zero_max_base_angular_speed_rad_s: float = 0.25
  stage_zero_max_hand_object_relative_speed_mps: float = 0.08


DEFAULT_GRASP_STATE_LIMITS = GraspStateDifficultyLimits()


@dataclass(frozen=True)
class GraspStateClassification:
  """Classifier output retained for logging and later boundary calibration."""

  stage: torch.Tensor
  difficulty: torch.Tensor
  primary_difficulty: torch.Tensor
  secondary_difficulty: torch.Tensor
  component_difficulties: torch.Tensor
  dominant_component: torch.Tensor
  stable_verified_grasp: torch.Tensor


def _validate_limits(limits: GraspStateDifficultyLimits) -> None:
  saturation_limits = (
    limits.axial_position_error_m,
    limits.lateral_position_error_m,
    limits.wrist_rotation_error_rad,
    limits.base_tilt_projected_gravity_xy,
    limits.base_linear_speed_mps,
    limits.base_angular_speed_rad_s,
    limits.hand_object_relative_speed_mps,
  )
  if any(value <= 0.0 for value in saturation_limits):
    raise ValueError("All grasp-state saturation limits must be positive.")
  if not 0.0 <= limits.secondary_difficulty_weight <= 1.0:
    raise ValueError("Secondary difficulty weight must lie inside [0, 1].")


def _validate_boundaries(boundaries: tuple[float, ...]) -> None:
  if len(boundaries) != 19:
    raise ValueError("Exactly 19 boundaries are required for Stage 1--20.")
  if not all(0.0 < value < 1.0 for value in boundaries):
    raise ValueError("Stage boundaries must lie strictly inside (0, 1).")
  if not all(
    left < right
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
  ):
    raise ValueError("Stage boundaries must be strictly increasing.")


def classify_grasp_state(
  features: GraspStateFeatures,
  limits: GraspStateDifficultyLimits = DEFAULT_GRASP_STATE_LIMITS,
  stage_boundaries: tuple[float, ...] = DEFAULT_GRASP_STAGE_BOUNDARIES,
) -> GraspStateClassification:
  """Classify every valid standing state into grasp-progress Stage 0--20.

  The worst normalized component is the primary difficulty. The mean of the
  remaining components adds a smaller bounded penalty, so a dangerous error is
  never averaged away while multiple simultaneous errors still make the state
  harder. Contact state applies a lower stage bound: zero contacts cannot be
  easier than Stage 3 and one-link contact cannot be easier than Stage 2. A
  strict grasp reaches Stage 0 only while the hand-object and base motion are
  inside the dedicated stability limits.
  """
  _validate_limits(limits)
  _validate_boundaries(stage_boundaries)

  values = torch.broadcast_tensors(
    features.axial_position_error_m,
    features.lateral_position_error_m,
    features.wrist_rotation_error_rad,
    features.gripper_open_fraction,
    features.joint_limit_risk,
    features.projected_gravity_xy,
    features.base_linear_speed_mps,
    features.base_angular_speed_rad_s,
    features.hand_object_relative_speed_mps,
    features.contact_link_count,
    features.verified_grasp,
  )
  (
    axial_error,
    lateral_error,
    wrist_error,
    gripper_open_fraction,
    joint_limit_risk,
    projected_gravity_xy,
    base_linear_speed,
    base_angular_speed,
    relative_speed,
    contact_link_count,
    verified_grasp,
  ) = values

  components = torch.stack(
    (
      torch.abs(axial_error) / limits.axial_position_error_m,
      torch.abs(lateral_error) / limits.lateral_position_error_m,
      torch.abs(wrist_error) / limits.wrist_rotation_error_rad,
      torch.abs(gripper_open_fraction),
      torch.abs(joint_limit_risk),
      torch.abs(projected_gravity_xy)
      / limits.base_tilt_projected_gravity_xy,
      torch.abs(base_linear_speed) / limits.base_linear_speed_mps,
      torch.abs(base_angular_speed) / limits.base_angular_speed_rad_s,
      torch.abs(relative_speed) / limits.hand_object_relative_speed_mps,
    ),
    dim=-1,
  ).clamp(min=0.0, max=1.0)
  primary_difficulty, dominant_component = components.max(dim=-1)
  secondary_difficulty = (
    components.sum(dim=-1) - primary_difficulty
  ) / (components.shape[-1] - 1)
  difficulty = primary_difficulty + (
    limits.secondary_difficulty_weight
    * (1.0 - primary_difficulty)
    * secondary_difficulty
  )

  boundaries = torch.tensor(
    stage_boundaries, device=difficulty.device, dtype=difficulty.dtype
  )
  stage = (torch.bucketize(difficulty.contiguous(), boundaries) + 1).to(
    dtype=torch.long
  )

  contact_link_count = contact_link_count.to(dtype=torch.long)
  minimum_contact_stage = torch.where(
    contact_link_count >= 2,
    torch.ones_like(stage),
    torch.where(
      contact_link_count == 1,
      torch.full_like(stage, 2),
      torch.full_like(stage, 3),
    ),
  )
  stage = torch.maximum(stage, minimum_contact_stage)

  stable_verified_grasp = (
    verified_grasp.to(dtype=torch.bool)
    & (contact_link_count >= 2)
    & (
      torch.abs(projected_gravity_xy)
      <= limits.stage_zero_max_projected_gravity_xy
    )
    & (
      torch.abs(base_linear_speed)
      <= limits.stage_zero_max_base_linear_speed_mps
    )
    & (
      torch.abs(base_angular_speed)
      <= limits.stage_zero_max_base_angular_speed_rad_s
    )
    & (
      torch.abs(relative_speed)
      <= limits.stage_zero_max_hand_object_relative_speed_mps
    )
  )
  stage = torch.where(stable_verified_grasp, torch.zeros_like(stage), stage)

  return GraspStateClassification(
    stage=stage,
    difficulty=difficulty,
    primary_difficulty=primary_difficulty,
    secondary_difficulty=secondary_difficulty,
    component_difficulties=components,
    dominant_component=dominant_component,
    stable_verified_grasp=stable_verified_grasp,
  )
