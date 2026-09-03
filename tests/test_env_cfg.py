import re

import pytest
from mjlab.envs.mdp.actions import JointPositionActionCfg
from talos_tabletop.tasks.tabletop.env_cfg import (
  FOOT_SWING_TARGET_HEIGHT_M,
  LOCOMOTION_PERIOD_S,
  MAX_COLLISION_OBSTACLES,
  ROBOT_SPAWN_POSITION_M,
  STATIONARY_GRASP_ROBOT_POSITION_M,
  talos_tabletop_grasp_env_cfg,
  talos_tabletop_position_tracking_env_cfg,
  talos_tabletop_reaching_env_cfg,
  talos_tabletop_stable_contact_lift_env_cfg,
  talos_tabletop_stationary_grasp_env_cfg,
)


def test_tabletop_cfg_has_independent_scene_and_upper_body_actions() -> None:
  cfg = talos_tabletop_reaching_env_cfg()

  assert set(cfg.scene.entities) == {"robot", "table", "object"}
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert action.actuator_names == (
    r"torso_.*_joint",
    r"arm_.*_joint",
    r"gripper_(left|right)_joint",
  )
  assert all("leg" not in expression for expression in action.actuator_names)
  assert {sensor.name for sensor in cfg.scene.sensors} == {
    "left_gripper_object_contact",
    "right_gripper_object_contact",
  }
  assert "object_pos_b" in cfg.observations["actor"].terms
  assert "lower_body_pose" in cfg.rewards


def test_grasp_cfg_uses_full_body_balance_and_privileged_object_center() -> None:
  cfg = talos_tabletop_grasp_env_cfg(object_shape="cube")

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert action.actuator_names == (
    r"leg_.*_joint",
    r"torso_.*_joint",
    r"arm_.*_joint",
    r"gripper_right_joint",
  )
  assert cfg.commands == {}
  assert "privileged_object_center_b" in cfg.observations["actor"].terms
  assert "privileged_object_center_b" in cfg.observations["critic"].terms
  assert "collision_obstacle_boxes_b" in cfg.observations["actor"].terms
  assert "table_approach_target_b" in cfg.observations["actor"].terms
  assert "placement_target_b" in cfg.observations["actor"].terms
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms
  assert cfg.observations["actor"].terms["locomotion_phase"].params == {
    "period": LOCOMOTION_PERIOD_S,
    "control_mode": "position_tracking",
  }
  assert {sensor.name for sensor in cfg.scene.sensors} == {
    "right_gripper_object_contact",
    "feet_ground_contact",
    "body_ground_contact",
    "body_table_contact",
  }
  assert {
    "standing_success",
    "navigate_to_table",
    "navigation_speed",
    "gait_contact",
    "swing_peak_height",
    "foot_slip",
    "face_table",
    "healthy_gait_success",
    "reach_table_success",
    "approach_object",
    "multi_link_contact",
    "lift_progress",
    "object_target",
    "place_success",
  } <= set(cfg.rewards)
  assert cfg.scene.entities["robot"].init_state.pos == pytest.approx(
    ROBOT_SPAWN_POSITION_M
  )
  assert cfg.scene.env_spacing == 0.0
  assert "reset_robot_root" in cfg.events
  assert "reset_table_root" in cfg.events
  assert cfg.events["reset_table_root"].params["pose_range"] == {}
  assert cfg.events["reset_robot_joints"].params["position_range"] == (0.0, 0.0)
  assert cfg.rewards["approach_object"].weight == 0.0
  assert cfg.rewards["multi_link_contact"].weight == 0.0
  assert cfg.rewards["lift_progress"].weight == 0.0
  assert cfg.rewards["right_arm_pose"].weight == -0.5
  assert cfg.terminations["object_lost"].params["minimum_height"] == -10.0
  assert set(cfg.curriculum) == {"task_stage"}
  curriculum_params = cfg.curriculum["task_stage"].params
  assert curriculum_params["promotion_reward_names"] == (
    "standing_success",
    "reach_table_success",
    "grasp_lift_success",
  )
  assert curriculum_params["promotion_success_rates"] == (0.80, 0.70, 0.60)
  assert curriculum_params["initial_stage"] == 0
  assert len(curriculum_params["stage_reward_weights"]) == 4
  assert all("step" not in key for key in curriculum_params)
  navigation_weights = curriculum_params["stage_reward_weights"][1]
  assert navigation_weights["navigate_to_table"] == 0.0
  assert navigation_weights["navigation_progress"] == 8.0
  assert navigation_weights["navigation_speed"] == 1.0
  assert navigation_weights["both_feet_contact"] == 0.0
  assert navigation_weights["gait_contact"] == 0.5
  assert navigation_weights["swing_peak_height"] == 5.0
  assert navigation_weights["foot_slip"] == -0.5
  assert navigation_weights["face_table"] == 2.0
  assert navigation_weights["standing_success"] == 0.1
  assert navigation_weights["base_motion"] == 0.0
  assert navigation_weights["termination_penalty"] == -1000.0
  assert cfg.rewards["navigate_to_table"].params["std"] == 3.0
  assert cfg.rewards["navigation_speed"].params == {
    "target_position": pytest.approx((0.0, 0.0)),
    "heading_target_position": pytest.approx((0.75, 0.0)),
    "minimum_speed": 0.05,
    "target_speed": 0.20,
    "maximum_projected_gravity_xy": 0.25,
    "sensor_name": "feet_ground_contact",
    "minimum_height": 0.85,
  }

  obstacle_params = cfg.observations["actor"].terms["collision_obstacle_boxes_b"].params
  assert obstacle_params["obstacle_names"] == ("table",)
  assert obstacle_params["max_obstacles"] == MAX_COLLISION_OBSTACLES

  body_ground = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "body_ground_contact"
  )
  assert re.fullmatch(body_ground.primary.pattern, "leg_left_4_link")
  assert not re.fullmatch(body_ground.primary.pattern, "leg_left_6_link")
  feet_ground = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "feet_ground_contact"
  )
  assert feet_ground.track_air_time
  assert cfg.rewards["swing_peak_height"].params["target_height"] == pytest.approx(
    FOOT_SWING_TARGET_HEIGHT_M
  )
  healthy_gait_params = cfg.rewards["healthy_gait_success"].params
  assert healthy_gait_params["minimum_swing_peak_height"] == pytest.approx(0.075)
  assert healthy_gait_params["maximum_swing_peak_height"] == pytest.approx(0.105)
  assert healthy_gait_params["gait_offsets"] == (0.0, 0.5)
  assert healthy_gait_params["minimum_gait_contact_match"] == pytest.approx(0.75)


def test_position_tracking_task_stops_before_manipulation_mode() -> None:
  cfg = talos_tabletop_position_tracking_env_cfg()
  params = cfg.curriculum["task_stage"].params

  assert params["initial_stage"] == 0
  assert params["promotion_reward_names"] == ("healthy_gait_success",)
  assert params["promotion_success_rates"] == (0.70,)
  assert len(params["stage_reward_weights"]) == 2
  gait_weights, tracking_weights = params["stage_reward_weights"]
  assert gait_weights["healthy_gait_success"] == 10.0
  assert gait_weights["reach_table_success"] == 0.0
  assert tracking_weights["reach_table_success"] == 10.0
  assert tracking_weights["approach_object"] == 0.0
  assert cfg.observations["actor"].terms["locomotion_phase"].params[
    "control_mode"
  ] == "position_tracking"


def test_stationary_grasp_task_has_verified_pick_curriculum() -> None:
  cfg = talos_tabletop_stationary_grasp_env_cfg()
  params = cfg.curriculum["task_stage"].params

  assert cfg.scene.entities["robot"].init_state.pos == pytest.approx(
    STATIONARY_GRASP_ROBOT_POSITION_M
  )
  assert params["promotion_reward_names"] == (
    "standing_success",
    "grasp_ready_success",
    "low_lift_success",
  )
  assert params["promotion_success_rates"] == (0.80, 0.45, 0.40)
  assert len(params["stage_reward_weights"]) == 4
  standing_weights, grasp_weights, micro_pick_weights, full_pick_weights = params[
    "stage_reward_weights"
  ]
  assert standing_weights["standing_success"] == 10.0
  assert standing_weights["approach_progress"] == 0.0
  assert grasp_weights["approach_progress"] == 4.0
  assert grasp_weights["grasp_quality_progress"] == 12.0
  assert grasp_weights["grasp_ready_success"] == 25.0
  assert grasp_weights["face_contact_area"] == 0.0
  assert grasp_weights["force_closure"] == 0.0
  assert grasp_weights["lift_progress"] == 0.0
  assert micro_pick_weights["lift_progress"] == 30.0
  assert micro_pick_weights["low_lift_success"] == 50.0
  assert full_pick_weights["lift_progress"] == 50.0
  assert full_pick_weights["grasp_lift_success"] == 80.0
  assert full_pick_weights["excessive_object_height"] == -20.0
  for weights in params["stage_reward_weights"]:
    assert weights["navigation_progress"] == 0.0
    assert weights["reach_table_success"] == 0.0
    assert weights["object_target"] == 0.0
    assert weights["place_success"] == 0.0
  assert params["stage_event_params"][0]["reset_table_root"]["pose_range"] == {
    "x": (10.0, 10.0)
  }
  assert params["stage_event_params"][1]["reset_table_root"]["pose_range"] == {}
  assert len(params["stage_event_params"]) == 4
  assert len(params["stage_termination_params"]) == 4
  assert cfg.rewards["standing_success"].params["maximum_linear_speed"] == 0.05
  assert cfg.rewards["standing_success"].params["maximum_angular_speed"] == 0.10
  assert cfg.observations["actor"].terms["locomotion_phase"].params[
    "control_mode"
  ] == "manipulation"
  assert cfg.observations["actor"].terms[
    "privileged_object_center_b"
  ].params["minimum_curriculum_stage"] == 1
  assert cfg.scene.sensors[0].fields == ("found", "force", "pos", "normal")
  assert cfg.scene.sensors[0].track_air_time is True
  assert cfg.rewards["low_lift_success"].params["required_duration_s"] == 0.50
  assert cfg.rewards["grasp_lift_success"].params["required_duration_s"] == 1.0


def test_stable_contact_lift_is_single_stage_and_checkpoint_compatible() -> None:
  cfg = talos_tabletop_stable_contact_lift_env_cfg()

  assert cfg.curriculum == {}
  assert cfg.scene.num_envs == 2048
  assert cfg.events["reset_object_on_pickup_zone"].params["pose_range"] == {}
  assert cfg.scene.entities["robot"].init_state.joint_pos[
    "gripper_right_joint"
  ] == pytest.approx(0.0)
  assert cfg.observations["actor"].terms[
    "privileged_object_center_b"
  ].params == {}
  assert "minimum_curriculum_stage" not in cfg.observations["actor"].terms[
    "collision_obstacle_boxes_b"
  ].params
  assert cfg.rewards["contact_lift_progress"].weight == 40.0
  assert cfg.rewards["contact_lift_hold"].weight == 12.0
  assert cfg.rewards["contact_lift_hold"].params["target_lift_height"] == 0.06
  assert cfg.rewards["contact_lift_hold"].params["minimum_contact_links"] == 2
  assert cfg.rewards["contact_lift_success"].weight == 80.0
  assert cfg.rewards["lift_progress"].weight == 0.0
