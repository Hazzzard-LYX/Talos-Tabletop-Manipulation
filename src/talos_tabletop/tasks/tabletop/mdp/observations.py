from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")

# Collision observations use a fixed-size oriented-box interface so future
# perception modules can populate predicted obstacles without changing the
# actor input size.  Each slot is:
#   center_b xyz, half_extents xyz, orientation_b wxyz, occupied flag.
OBSTACLE_BOX_FEATURE_DIM = 11


def locomotion_phase(
  env: ManagerBasedRlEnv,
  period: float,
  control_mode: Literal["position_tracking", "manipulation"],
) -> torch.Tensor:
  """Encode the walking clock and reserve an explicit stationary mode.

  Position tracking uses a unit-circle phase signal.  Manipulation returns the
  otherwise unreachable zero vector, so the same network interface can later
  represent a stationary manipulation controller without adding another
  constant observation that would collapse under running normalization.
  """
  if period <= 0.0:
    raise ValueError("locomotion period must be positive")
  if control_mode == "manipulation":
    return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float)
  if control_mode != "position_tracking":
    raise ValueError(f"Unsupported control mode: {control_mode}")

  phase = (env.episode_length_buf * env.step_dt) % period / period
  angle = 2.0 * torch.pi * phase
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)


def commands_gen(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)

  des_pos_b = command.command

  return des_pos_b


def object_position_in_robot_frame(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Return the object position relative to the robot root frame."""
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  offset_w = obj.data.root_link_pos_w - robot.data.root_link_pos_w
  return quat_apply_inverse(robot.data.root_link_quat_w, offset_w)


def object_vector_from_site(
  env: ManagerBasedRlEnv,
  site_name: str,
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Return the world-frame vector from a robot site to the object center."""
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  site_id = robot.site_names.index(site_name)
  return obj.data.root_link_pos_w - robot.data.site_pos_w[:, site_id]


def point_in_robot_frame(
  env: ManagerBasedRlEnv,
  point: tuple[float, float, float],
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return an environment-local target point in the robot root frame."""
  robot: Entity = env.scene[robot_cfg.name]
  point_w = torch.tensor(point, device=env.device, dtype=torch.float).repeat(
    env.num_envs, 1
  )
  point_w += env.scene.env_origins
  return quat_apply_inverse(
    robot.data.root_link_quat_w,
    point_w - robot.data.root_link_pos_w,
  )


def obstacle_boxes_in_robot_frame(
  env: ManagerBasedRlEnv,
  obstacle_names: tuple[str, ...],
  half_extents: tuple[tuple[float, float, float], ...],
  max_obstacles: int,
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Encode collision obstacles as fixed-capacity oriented boxes.

  The returned flattened tensor contains ``max_obstacles`` slots of 11 values:
  ``center_b[3], half_extents[3], orientation_b[4], occupied[1]``.  Unused
  slots are zero.  A future collision predictor can fill those reserved slots
  with arbitrary world obstacles while preserving the observation contract.
  """
  if len(obstacle_names) != len(half_extents):
    raise ValueError("obstacle_names and half_extents must have equal length.")
  if len(obstacle_names) > max_obstacles:
    raise ValueError("Configured obstacles exceed max_obstacles capacity.")

  robot: Entity = env.scene[robot_cfg.name]
  descriptors = torch.zeros(
    (env.num_envs, max_obstacles, OBSTACLE_BOX_FEATURE_DIM),
    device=env.device,
    dtype=torch.float,
  )
  robot_quat_inv = quat_conjugate(robot.data.root_link_quat_w)

  for slot, (name, extent) in enumerate(zip(obstacle_names, half_extents, strict=True)):
    obstacle: Entity = env.scene[name]
    offset_w = obstacle.data.root_link_pos_w - robot.data.root_link_pos_w
    descriptors[:, slot, 0:3] = quat_apply_inverse(
      robot.data.root_link_quat_w, offset_w
    )
    descriptors[:, slot, 3:6] = torch.tensor(
      extent, device=env.device, dtype=torch.float
    )
    descriptors[:, slot, 6:10] = quat_mul(
      robot_quat_inv, obstacle.data.root_link_quat_w
    )
    descriptors[:, slot, 10] = 1.0

  return descriptors.flatten(start_dim=1)
