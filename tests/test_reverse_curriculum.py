import math

import pytest
import torch
from talos_tabletop.tasks.tabletop.mdp.reverse_curriculum import (
  DEFAULT_REVERSE_CURRICULUM_STAGES,
  GRASP_DIFFICULTY_COMPONENT_NAMES,
  GraspStateDifficultyLimits,
  GraspStateFeatures,
  assess_reverse_curriculum_promotion,
  build_reverse_curriculum_stage_definitions,
  classify_grasp_state,
  talos_gripper_open_fraction,
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


def test_talos_gripper_joint_sign_maps_closed_and_open_correctly() -> None:
  positions = torch.tensor([-0.959931, -0.4799655, 0.0])
  assert torch.allclose(
    talos_gripper_open_fraction(positions),
    torch.tensor([0.0, 0.5, 1.0]),
  )


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


def test_reverse_curriculum_defines_all_twenty_training_stages() -> None:
  assert tuple(stage.stage for stage in DEFAULT_REVERSE_CURRICULUM_STAGES) == tuple(
    range(1, 21)
  )

  for definition in DEFAULT_REVERSE_CURRICULUM_STAGES[:-1]:
    stage = definition.stage
    assert definition.initialization_mixture == (
      (stage - 1, 0.30),
      (stage, 0.60),
      (stage + 1, 0.10),
    )
    assert sum(probability for _, probability in definition.initialization_mixture) == (
      pytest.approx(1.0)
    )
    assert definition.promotion_success_rate == pytest.approx(0.80)
    assert definition.promotion_evaluation_stage == stage
    assert definition.evaluation_episodes == 4096
    assert definition.success_hold_duration_s == pytest.approx(5.0)
    assert definition.success_lift_height_m == pytest.approx(0.06)


def test_final_stage_uses_seventy_thirty_mixture_without_promotion() -> None:
  definition = DEFAULT_REVERSE_CURRICULUM_STAGES[-1]
  assert definition.initialization_mixture == ((19, 0.30), (20, 0.70))
  assert definition.promotion_success_rate is None
  assert definition.promotion_evaluation_stage is None
  assert definition.evaluation_episodes is None


def test_promotion_requires_more_than_eighty_percent_in_full_window() -> None:
  definition = build_reverse_curriculum_stage_definitions(
    evaluation_episodes=100
  )[4]

  success_rate, ready = assess_reverse_curriculum_promotion(
    successful_current_stage_episodes=80,
    completed_current_stage_episodes=100,
    definition=definition,
  )
  assert success_rate == pytest.approx(0.80)
  assert ready is False

  success_rate, ready = assess_reverse_curriculum_promotion(
    successful_current_stage_episodes=81,
    completed_current_stage_episodes=100,
    definition=definition,
  )
  assert success_rate == pytest.approx(0.81)
  assert ready is True


def test_easier_and_harder_replay_do_not_enter_promotion_rate() -> None:
  definition = build_reverse_curriculum_stage_definitions(
    evaluation_episodes=10
  )[9]
  assert definition.promotion_evaluation_stage == 10
  assert definition.initialization_probability(9) == pytest.approx(0.30)
  assert definition.initialization_probability(10) == pytest.approx(0.60)
  assert definition.initialization_probability(11) == pytest.approx(0.10)
  assert definition.initialization_probability(12) == pytest.approx(0.0)

  _, ready = assess_reverse_curriculum_promotion(
    successful_current_stage_episodes=8,
    completed_current_stage_episodes=10,
    definition=definition,
  )
  assert ready is False
