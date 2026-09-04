"""Task-local reset events with MJLab 1.5 broadcasting compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _rows_for_envs(value: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
  """Select batched rows, broadcasting MJLab's singleton static data."""
  if value.shape[0] == 1:
    return value.expand(len(env_ids), *value.shape[1:])
  return value[env_ids]


def _sample_se3_range(
  ranges: dict[str, tuple[float, float]] | None,
  count: int,
  device: str,
) -> torch.Tensor:
  """Sample xyz/roll-pitch-yaw offsets without depending on MJLab internals."""
  ranges = ranges or {}
  columns = []
  for key in ("x", "y", "z", "roll", "pitch", "yaw"):
    low, high = ranges.get(key, (0.0, 0.0))
    columns.append(sample_uniform(low, high, (count,), device))
  return torch.stack(columns, dim=-1)


class reset_from_anchor_bank:
  """Reset robot and object state from a validated reverse-curriculum bank.

  The bank is loaded once when the environment is constructed and then kept on
  the simulator device.  Every reset samples anchors uniformly, which is also
  uniform over stages for the generated 20-stage bank (40 anchors per stage).
  Domain-randomization ranges are event parameters so the performance
  curriculum can strengthen them without rebuilding the environment.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    bank_path = Path(cfg.params["anchor_bank_path"]).expanduser()
    if not bank_path.is_file():
      raise FileNotFoundError(
        f"Standing anchor bank does not exist: {bank_path}. Set "
        "TALOS_STANDING_ANCHOR_BANK to the generated .pt file."
      )

    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    anchors = bank.get("anchors", ())
    if not anchors:
      raise ValueError(f"Standing anchor bank contains no anchors: {bank_path}")

    robot: Entity = env.scene[cfg.params["robot_cfg"].name]
    bank_joint_names = tuple(bank.get("joint_names", ()))
    if bank_joint_names != tuple(robot.joint_names):
      raise ValueError(
        "Anchor-bank joint ordering differs from the current TALOS model: "
        f"bank={len(bank_joint_names)} joints, model={robot.num_joints} joints."
      )

    self._robot_root_state = self._stack_anchor_field(
      anchors, "robot_root_state", 13, env.device
    )
    self._robot_joint_position = self._stack_anchor_field(
      anchors, "robot_joint_position", robot.num_joints, env.device
    )
    self._robot_joint_velocity = self._stack_anchor_field(
      anchors, "robot_joint_velocity", robot.num_joints, env.device
    )
    self._object_root_state = self._stack_anchor_field(
      anchors, "object_root_state", 13, env.device
    )
    self._anchor_stages = torch.tensor(
      [int(anchor["requested_stage"]) for anchor in anchors],
      dtype=torch.long,
      device=env.device,
    )
    self._num_anchors = len(anchors)
    self._randomized_joint_ids = torch.tensor(
      [
        index
        for index, name in enumerate(robot.joint_names)
        if name.startswith(("leg_", "torso_", "arm_")) or name == "gripper_right_joint"
      ],
      dtype=torch.long,
      device=env.device,
    )
    env._talos_anchor_indices = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._talos_anchor_stages = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  @staticmethod
  def _stack_anchor_field(
    anchors: list[dict], field: str, width: int, device: str
  ) -> torch.Tensor:
    values = torch.stack(
      [torch.as_tensor(anchor[field], dtype=torch.float32) for anchor in anchors]
    )
    if values.shape != (len(anchors), width):
      raise ValueError(
        f"Anchor field {field!r} has shape {tuple(values.shape)}, "
        f"expected ({len(anchors)}, {width})."
      )
    if not torch.isfinite(values).all():
      raise ValueError(f"Anchor field {field!r} contains non-finite values.")
    return values.to(device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    anchor_bank_path: str,
    joint_position_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
    root_pose_range: dict[str, tuple[float, float]],
    root_velocity_range: dict[str, tuple[float, float]],
    object_pose_range: dict[str, tuple[float, float]],
    object_velocity_range: dict[str, tuple[float, float]],
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
  ) -> None:
    del anchor_bank_path
    env_ids = resolve_env_ids(env, env_ids)
    count = len(env_ids)
    anchor_ids = torch.randint(
      self._num_anchors, (count,), dtype=torch.long, device=env.device
    )
    robot: Entity = env.scene[robot_cfg.name]
    object_entity: Entity = env.scene[object_cfg.name]

    robot_root = self._robot_root_state[anchor_ids].clone()
    root_offset = _sample_se3_range(root_pose_range, count, env.device)
    robot_root[:, :3] += root_offset[:, :3] + env.scene.env_origins[env_ids]
    root_rotation = quat_from_euler_xyz(
      root_offset[:, 3], root_offset[:, 4], root_offset[:, 5]
    )
    robot_root[:, 3:7] = quat_mul(robot_root[:, 3:7], root_rotation)
    robot_root[:, 7:13] += _sample_se3_range(root_velocity_range, count, env.device)

    joint_position = self._robot_joint_position[anchor_ids].clone()
    randomized_positions = joint_position[:, self._randomized_joint_ids]
    randomized_positions += sample_uniform(
      *joint_position_range, randomized_positions.shape, env.device
    )
    joint_position[:, self._randomized_joint_ids] = randomized_positions
    soft_limits = robot.data.soft_joint_pos_limits
    if soft_limits is None:
      raise RuntimeError("Robot soft joint limits are unavailable during reset.")
    joint_limits = _rows_for_envs(soft_limits, env_ids)
    joint_position.clamp_(joint_limits[..., 0], joint_limits[..., 1])

    joint_velocity = self._robot_joint_velocity[anchor_ids].clone()
    randomized_velocities = joint_velocity[:, self._randomized_joint_ids]
    randomized_velocities += sample_uniform(
      *joint_velocity_range, randomized_velocities.shape, env.device
    )
    joint_velocity[:, self._randomized_joint_ids] = randomized_velocities

    object_root = self._object_root_state[anchor_ids].clone()
    object_offset = _sample_se3_range(object_pose_range, count, env.device)
    object_root[:, :3] += object_offset[:, :3] + env.scene.env_origins[env_ids]
    object_rotation = quat_from_euler_xyz(
      object_offset[:, 3], object_offset[:, 4], object_offset[:, 5]
    )
    object_root[:, 3:7] = quat_mul(object_root[:, 3:7], object_rotation)
    object_root[:, 7:13] += _sample_se3_range(object_velocity_range, count, env.device)

    robot.write_root_link_pose_to_sim(robot_root[:, :7], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(robot_root[:, 7:13], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
    object_entity.write_root_link_pose_to_sim(object_root[:, :7], env_ids=env_ids)
    object_entity.write_root_link_velocity_to_sim(object_root[:, 7:13], env_ids=env_ids)

    env._talos_anchor_indices[env_ids] = anchor_ids
    env._talos_anchor_stages[env_ids] = self._anchor_stages[anchor_ids]


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
