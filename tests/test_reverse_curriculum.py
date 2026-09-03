import math

import pytest
import torch
from talos_tabletop.tasks.tabletop.mdp.reverse_curriculum import (
  GRASP_DIFFICULTY_COMPONENT_NAMES,
  GraspStateDifficultyLimits,
  GraspStateFeatures,
  classify_grasp_state,
)


def _features(batch_size: int, **overrides: torch.Tensor) -> GraspStateFeatures:
  defaults = {
    "axial_position_error_m": torch.zeros(batch_size),
    "lateral_position_error_m": torch.zeros(batch_size),
    "wrist_rotation_error_rad": torch.zeros(batch_size),
    "gripper_open_fraction": torch.zeros(batch_size),
    "joint_limit_risk": torch.zeros(batch_size),
    "projected_gravity_xy": torch.zeros(batch_size),
    "base_linear_speed_mps": torch.zeros(batch_size),
    "base_angular_speed_rad_s": torch.zeros(batch_size),
    "hand_object_relative_speed_mps": torch.zeros(batch_size),
    "contact_link_count": torch.full((batch_size,), 2),
    "verified_grasp": torch.zeros(batch_size, dtype=torch.bool),
  }
  defaults.update(overrides)
  return GraspStateFeatures(**defaults)


def test_all_twenty_ungrasped_stages_are_reachable() -> None:
  normalized = (torch.arange(20, dtype=torch.float) + 0.5) / 20.0
  result = classify_grasp_state(
    _features(20, axial_position_error_m=normalized * 0.30)
  )
  assert result.stage.tolist() == list(range(1, 21))


def test_verified_stable_grasp_is_stage_zero() -> None:
  result = classify_grasp_state(
    _features(
      1,
      verified_grasp=torch.tensor([True]),
      contact_link_count=torch.tensor([3]),
      hand_object_relative_speed_mps=torch.tensor([0.05]),
    )
  )
  assert result.stage.item() == 0
  assert result.stable_verified_grasp.item() is True


def test_unstable_grasp_does_not_enter_stage_zero() -> None:
  result = classify_grasp_state(
    _features(
      1,
      verified_grasp=torch.tensor([True]),
      contact_link_count=torch.tensor([3]),
      hand_object_relative_speed_mps=torch.tensor([0.20]),
    )
  )
  assert result.stage.item() > 0
  assert result.stable_verified_grasp.item() is False


def test_contact_state_sets_minimum_ungrasped_stage() -> None:
  result = classify_grasp_state(
    _features(
      3,
      contact_link_count=torch.tensor([2, 1, 0]),
    )
  )
  assert result.stage.tolist() == [1, 2, 3]


def test_worst_component_dominates_but_secondary_errors_contribute() -> None:
  result = classify_grasp_state(
    _features(
      1,
      axial_position_error_m=torch.tensor([0.03]),
      lateral_position_error_m=torch.tensor([0.02]),
      wrist_rotation_error_rad=torch.tensor([0.1 * math.pi]),
      gripper_open_fraction=torch.tensor([0.75]),
    )
  )
  assert result.stage.item() == 16
  assert result.primary_difficulty.item() == pytest.approx(0.75)
  assert result.secondary_difficulty.item() == pytest.approx(0.0375)
  assert result.difficulty.item() == pytest.approx(0.75328125)
  assert GRASP_DIFFICULTY_COMPONENT_NAMES[result.dominant_component.item()] == (
    "gripper_opening"
  )


def test_same_maximum_is_harder_with_multiple_secondary_errors() -> None:
  single_error = classify_grasp_state(
    _features(1, gripper_open_fraction=torch.tensor([0.50]))
  )
  multiple_errors = classify_grasp_state(
    _features(
      1,
      axial_position_error_m=torch.tensor([0.15]),
      lateral_position_error_m=torch.tensor([0.10]),
      wrist_rotation_error_rad=torch.tensor([0.50 * math.pi]),
      gripper_open_fraction=torch.tensor([0.50]),
      joint_limit_risk=torch.tensor([0.50]),
    )
  )

  assert single_error.primary_difficulty.item() == pytest.approx(0.50)
  assert multiple_errors.primary_difficulty.item() == pytest.approx(0.50)
  assert multiple_errors.secondary_difficulty.item() == pytest.approx(0.25)
  assert multiple_errors.difficulty.item() == pytest.approx(0.54375)
  assert multiple_errors.difficulty.item() > single_error.difficulty.item()
  assert multiple_errors.stage.item() > single_error.stage.item()


def test_extreme_valid_state_saturates_at_stage_twenty() -> None:
  result = classify_grasp_state(
    _features(1, lateral_position_error_m=torch.tensor([10.0]))
  )
  assert result.stage.item() == 20
  assert result.difficulty.item() == 1.0


def test_custom_boundaries_are_supported_for_later_calibration() -> None:
  boundaries = tuple(0.01 * index for index in range(1, 20))
  result = classify_grasp_state(
    _features(1, axial_position_error_m=torch.tensor([0.045])),
    stage_boundaries=boundaries,
  )
  assert result.stage.item() == 15


def test_invalid_boundary_count_is_rejected() -> None:
  with pytest.raises(ValueError):
    classify_grasp_state(_features(1), stage_boundaries=(0.5,))


def test_invalid_secondary_weight_is_rejected() -> None:
  with pytest.raises(ValueError):
    classify_grasp_state(
      _features(1),
      limits=GraspStateDifficultyLimits(secondary_difficulty_weight=1.01),
    )
