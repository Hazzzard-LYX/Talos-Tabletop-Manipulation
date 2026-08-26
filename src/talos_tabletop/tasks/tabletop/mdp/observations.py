from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


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
