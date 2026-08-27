"""Task-local reset events with MJLab 1.5 broadcasting compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _rows_for_envs(value: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
  """Select batched rows, broadcasting MJLab's singleton static data."""
  if value.shape[0] == 1:
    return value.expand(len(env_ids), *value.shape[1:])
  return value[env_ids]


def reset_joints_by_offset_broadcast(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joints while supporting singleton soft-limit tensors.

  MJLab 1.5 stores static soft joint limits with a leading size of one, but its
  stock reset event indexes that dimension with environment IDs.  This local
  event broadcasts singleton state before selection, so command-line overrides
  such as ``--env.scene.num-envs 256`` reset every environment correctly.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]

  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  if default_joint_pos is None or default_joint_vel is None:
    raise RuntimeError("Robot default joint state is unavailable during reset.")
  if soft_joint_pos_limits is None:
    raise RuntimeError("Robot soft joint limits are unavailable during reset.")

  joint_ids = asset_cfg.joint_ids
  joint_pos = _rows_for_envs(default_joint_pos, env_ids)[:, joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_pos_limits = _rows_for_envs(soft_joint_pos_limits, env_ids)[:, joint_ids]
  joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = _rows_for_envs(default_joint_vel, env_ids)[:, joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  write_joint_ids = joint_ids
  if isinstance(write_joint_ids, list):
    write_joint_ids = torch.tensor(write_joint_ids, device=env.device)
  asset.write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    env_ids=env_ids,
    joint_ids=write_joint_ids,
  )
