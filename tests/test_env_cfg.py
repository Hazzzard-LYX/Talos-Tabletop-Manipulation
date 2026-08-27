import re

import pytest
from mjlab.envs.mdp.actions import JointPositionActionCfg
from talos_tabletop.tasks.tabletop.env_cfg import (
  MAX_COLLISION_OBSTACLES,
  ROBOT_SPAWN_POSITION_M,
  talos_tabletop_grasp_env_cfg,
  talos_tabletop_reaching_env_cfg,
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
  assert navigation_weights["navigation_progress"] == 15.0
  assert navigation_weights["navigation_speed"] == 4.0
  assert navigation_weights["standing_success"] == 0.05
  assert navigation_weights["base_motion"] == 0.0
  assert navigation_weights["termination_penalty"] == -500.0
  assert cfg.rewards["navigate_to_table"].params["std"] == 3.0
  assert cfg.rewards["navigation_speed"].params == {
    "target_position": pytest.approx((0.0, 0.0)),
    "minimum_speed": 0.05,
    "target_speed": 0.25,
    "maximum_projected_gravity_xy": 0.70,
  }

  obstacle_params = cfg.observations["actor"].terms["collision_obstacle_boxes_b"].params
  assert obstacle_params["obstacle_names"] == ("table",)
  assert obstacle_params["max_obstacles"] == MAX_COLLISION_OBSTACLES

  body_ground = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "body_ground_contact"
  )
  assert re.fullmatch(body_ground.primary.pattern, "leg_left_4_link")
  assert not re.fullmatch(body_ground.primary.pattern, "leg_left_6_link")
