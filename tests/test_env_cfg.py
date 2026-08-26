import re

from mjlab.envs.mdp.actions import JointPositionActionCfg
from talos_tabletop.tasks.tabletop.env_cfg import (
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
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms
  assert {sensor.name for sensor in cfg.scene.sensors} == {
    "right_gripper_object_contact",
    "feet_ground_contact",
    "body_ground_contact",
    "body_table_contact",
  }
  assert {"approach_object", "multi_link_contact", "lift_progress"} <= set(
    cfg.rewards
  )

  body_ground = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "body_ground_contact"
  )
  assert re.fullmatch(body_ground.primary.pattern, "leg_left_4_link")
  assert not re.fullmatch(body_ground.primary.pattern, "leg_left_6_link")
