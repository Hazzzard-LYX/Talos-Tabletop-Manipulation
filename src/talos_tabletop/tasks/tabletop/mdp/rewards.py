from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_conjugate, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def position_command_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  site_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)

  des_pos_b = command.command[:, :3]

  root_pos_w = asset.data.site_pos_w[:, 0]  # Root site position
  root_quat_w = asset.data.site_quat_w[:, 0]  # Root site quaternion

  # Transform position: p_w = p_root + R_root * p_b
  pos_rotated = quat_mul(
    quat_mul(
      root_quat_w,
      torch.cat([torch.zeros(env.num_envs, 1, device=env.device), des_pos_b], dim=1),
    ),
    quat_conjugate(root_quat_w),
  )[:, 1:]  # Extract xyz from quaternion product
  des_pos_w = root_pos_w + pos_rotated

  current_site_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]

  pos_error = current_site_pos_w - des_pos_w

  return torch.norm(pos_error, dim=1)


def base_motion_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize root linear and angular motion while manipulating in place."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_vel_w), dim=1)


def joint_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize selected joints moving away from their default standing pose."""
  asset: Entity = env.scene[asset_cfg.name]
  error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.square(error), dim=1)


def site_object_distance_tanh(
  env: ManagerBasedRlEnv,
  site_name: str,
  std: float,
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Dense approach reward from a hand grasp center to the object."""
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  site_id = robot.site_names.index(site_name)
  distance = torch.linalg.vector_norm(
    robot.data.site_pos_w[:, site_id] - obj.data.root_link_pos_w,
    dim=1,
  )
  return 1.0 - torch.tanh(distance / std)


def gripper_object_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  contacts_for_full_reward: int = 2,
) -> torch.Tensor:
  """Reward multi-link hand contact with the object.

  Requiring at least two independently matched gripper bodies discourages the
  policy from treating a single fingertip touch as a completed grasp.
  """
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
    raise TypeError(f"'{sensor_name}' must provide ContactSensor found data.")
  contacts = sensor.data.found.float().sum(dim=1)
  env.extras["log"]["Metrics/gripper_object_contacts"] = contacts.mean()
  return (contacts / float(contacts_for_full_reward)).clamp(max=1.0)


def object_lift_progress(
  env: ManagerBasedRlEnv,
  initial_center_height: float,
  target_lift_height: float,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Return normalized upward progress from the object's tabletop pose."""
  obj: Entity = env.scene[object_cfg.name]
  lift = obj.data.root_link_pos_w[:, 2] - initial_center_height
  progress = (lift / target_lift_height).clamp(min=0.0, max=1.0)
  env.extras["log"]["Metrics/object_lift_height_m"] = lift.mean()
  return progress


def grasp_and_lift_success(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  initial_center_height: float,
  minimum_lift_height: float,
  minimum_contacts: int = 2,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Reward the transition state needed by the later transport policy."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
    raise TypeError(f"'{sensor_name}' must provide ContactSensor found data.")
  obj: Entity = env.scene[object_cfg.name]
  enough_contacts = sensor.data.found.sum(dim=1) >= minimum_contacts
  high_enough = (
    obj.data.root_link_pos_w[:, 2] - initial_center_height
  ) >= minimum_lift_height
  success = enough_contacts & high_enough
  env.extras["log"]["Metrics/grasp_lift_success"] = success.float().mean()
  return success.float()


def base_position_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize horizontal base drift from the standing spawn location."""
  asset: Entity = env.scene[asset_cfg.name]
  spawn_xy = env.scene.env_origins[:, :2] + asset.data.default_root_state[:, :2]
  return torch.sum(torch.square(asset.data.root_link_pos_w[:, :2] - spawn_xy), dim=1)


def both_feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Reward retaining ground contact on both feet while manipulating."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
    raise TypeError(f"'{sensor_name}' must provide ContactSensor found data.")
  found = sensor.data.found > 0
  num_slots = sensor.cfg.num_slots
  per_foot = found.reshape(env.num_envs, len(sensor.primary_names), num_slots).any(dim=2)
  return per_foot.all(dim=1).float()


def position_command_error_tanh(
  env: ManagerBasedRlEnv,
  command_name: str,
  site_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)

  des_pos_b = command.command[:, :3]

  root_pos_w = asset.data.site_pos_w[:, 0]  # Root site position
  root_quat_w = asset.data.site_quat_w[:, 0]  # Root site quaternion

  pos_rotated = quat_mul(
    quat_mul(
      root_quat_w,
      torch.cat([torch.zeros(env.num_envs, 1, device=env.device), des_pos_b], dim=1),
    ),
    quat_conjugate(root_quat_w),
  )[:, 1:]  # Extract xyz from quaternion product
  des_pos_w = root_pos_w + pos_rotated

  current_site_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]

  pos_error = current_site_pos_w - des_pos_w
  distance = torch.norm(pos_error, dim=1)

  return 1 - torch.tanh(distance / std)


class action_rate_l2_louis:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    self._joint_ids = [
      asset.actuator_names.index(jname)
      for jname in joint_names
      if jname in asset.actuator_names
    ]

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.sum(
      torch.square(
        env.action_manager.action[:, self._joint_ids]
        - env.action_manager.prev_action[:, self._joint_ids]
      ),
      dim=1,
    )


def orientation_command_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  site_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  # Get robot entity and command
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)

  # Desired orientation in base frame (qw, qx, qy, qz)
  des_quat_b = command.command[:, 3:]

  # Root (base) orientation in world frame
  root_quat_w = asset.data.site_quat_w[:, 0]

  # Transform desired orientation from base -> world:
  # q_des_w = q_root_w ⊗ q_des_b
  des_quat_w = quat_mul(root_quat_w, des_quat_b)

  # Current site orientation in world frame
  site_idx = asset.site_names.index(site_name)
  current_quat_w = asset.data.site_quat_w[:, site_idx]

  # Quaternion error magnitude (angle between quaternions)
  ori_error = quat_error_magnitude(des_quat_w, current_quat_w)

  return ori_error


def stand_still_joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  abs_error = torch.abs(error)

  # amount beyond the 0.1 margin
  excess = torch.relu(abs_error - 0.001)

  # per-env penalty: sum of excess across all monitored joints
  penalty = torch.sum(excess, dim=1)
  return penalty


def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.
  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)
