from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def root_height_below(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Terminate after the floating base has clearly fallen."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < minimum_height


def root_tilt_exceeded(
  env: ManagerBasedRlEnv,
  maximum_projected_gravity_xy: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Terminate when root tilt is outside the recoverable standing region."""
  asset: Entity = env.scene[asset_cfg.name]
  tilt_measure = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
  return tilt_measure > maximum_projected_gravity_xy


def object_height_below(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Terminate once an object pushed off the table is no longer recoverable."""
  obj: Entity = env.scene[object_cfg.name]
  return obj.data.root_link_pos_w[:, 2] < minimum_height
