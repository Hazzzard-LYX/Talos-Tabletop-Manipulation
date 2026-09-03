import math

import pytest
import torch
from talos_tabletop.tasks.tabletop.mdp.reverse_curriculum import (
  REVERSE_CURRICULUM_STAGES,
  classify_grasp_progress_stage,
  get_reverse_curriculum_stage,
)


def test_reverse_curriculum_defines_all_twenty_ordered_stages() -> None:
  assert [spec.stage for spec in REVERSE_CURRICULUM_STAGES] == list(range(1, 21))
  assert [spec.focus for spec in REVERSE_CURRICULUM_STAGES] == [
    "gripper_opening",
    "gripper_opening",
    "axial_retreat",
    "axial_retreat",
    "axial_retreat",
    "lateral_position",
    "lateral_position",
    "wrist_rotation",
    "wrist_rotation",
    "wrist_rotation",
    "arm_torso_joints",
    "arm_torso_joints",
    "base_state",
    "base_state",
    "object_geometry",
    "object_geometry",
    "object_geometry",
    "object_dynamics",
    "object_dynamics",
    "object_dynamics",
  ]


def test_reset_stage_limits_are_cumulative() -> None:
  scalar_fields = (
    "max_gripper_opening_delta_rad",
    "max_axial_retreat_m",
    "max_lateral_position_error_m",
    "max_wrist_rotation_error_rad",
    "max_arm_torso_joint_error_rad",
    "max_base_tilt_error_rad",
    "max_base_linear_speed_mps",
    "max_base_angular_speed_rad_s",
    "max_object_xy_error_m",
    "max_object_yaw_error_rad",
  )
  for previous, current in zip(
    REVERSE_CURRICULUM_STAGES[:-1],
    REVERSE_CURRICULUM_STAGES[1:],
    strict=True,
  ):
    for field in scalar_fields:
      assert getattr(current, field) >= getattr(previous, field)
    for field in (
      "object_size_scale_range",
      "object_mass_scale_range",
      "object_friction_scale_range",
    ):
      previous_range = getattr(previous, field)
      current_range = getattr(current, field)
      assert current_range[0] <= previous_range[0]
      assert current_range[1] >= previous_range[1]


def test_stage_lookup_rejects_stage_zero_and_out_of_range() -> None:
  assert get_reverse_curriculum_stage(1) is REVERSE_CURRICULUM_STAGES[0]
  assert get_reverse_curriculum_stage(20) is REVERSE_CURRICULUM_STAGES[-1]
  with pytest.raises(ValueError):
    get_reverse_curriculum_stage(0)
  with pytest.raises(ValueError):
    get_reverse_curriculum_stage(21)


def test_progress_classifier_reserves_zero_for_verified_grasp() -> None:
  stage = classify_grasp_progress_stage(
    gripper_opening_delta_rad=torch.tensor([10.0]),
    axial_position_error_m=torch.tensor([10.0]),
    lateral_position_error_m=torch.tensor([10.0]),
    wrist_rotation_error_rad=torch.tensor([math.pi]),
    grasp_formed=torch.tensor([True]),
  )
  assert stage.item() == 0


def test_progress_classifier_maps_ungrasped_states_to_one_through_twenty() -> None:
  stage = classify_grasp_progress_stage(
    gripper_opening_delta_rad=torch.tensor([0.0, 0.025, 0.10, 1.0]),
    axial_position_error_m=torch.zeros(4),
    lateral_position_error_m=torch.zeros(4),
    wrist_rotation_error_rad=torch.zeros(4),
    grasp_formed=torch.zeros(4, dtype=torch.bool),
  )
  assert stage.tolist() == [1, 5, 20, 20]


def test_progress_classifier_uses_worst_normalized_error() -> None:
  stage = classify_grasp_progress_stage(
    gripper_opening_delta_rad=torch.tensor([0.01]),
    axial_position_error_m=torch.tensor([0.02]),
    lateral_position_error_m=torch.tensor([0.075]),
    wrist_rotation_error_rad=torch.tensor([math.radians(9)]),
    grasp_formed=torch.tensor([False]),
  )
  # Lateral error is 75% of its saturation limit and therefore selects Stage 15.
  assert stage.item() == 15
