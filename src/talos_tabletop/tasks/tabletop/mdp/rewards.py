from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_error_magnitude,
  quat_mul,
)

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


def action_magnitude_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the applied normalized action, including constant saturation."""
  return torch.sum(torch.square(env.action_manager.action), dim=1)


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


class site_object_record_progress:
  """Reward only new within-episode records in hand-object approach quality."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._best = torch.zeros(env.num_envs, device=env.device)
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    site_name: str,
    std: float,
    robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  ) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    obj: Entity = env.scene[object_cfg.name]
    site_id = robot.site_names.index(site_name)
    distance = torch.linalg.vector_norm(
      robot.data.site_pos_w[:, site_id] - obj.data.root_link_pos_w,
      dim=1,
    )
    quality = 1.0 - torch.tanh(distance / std)
    improvement = torch.where(
      self._initialized,
      (quality - self._best).clamp(min=0.0),
      torch.zeros_like(quality),
    )
    self._best = torch.where(
      self._initialized, torch.maximum(self._best, quality), quality
    )
    self._initialized[:] = True
    env.extras["log"]["Metrics/approach_record_quality"] = self._best.mean()
    env.extras["log"]["Metrics/approach_record_improvement"] = improvement.mean()
    # MJLab integrates reward terms with step_dt.  Returning a rate makes the
    # accumulated episode contribution equal the dimensionless improvement.
    return improvement / env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._best[env_ids] = 0.0
    self._initialized[env_ids] = False


def task_time(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Unit-rate term for a small task-completion time penalty."""
  return torch.ones(env.num_envs, device=env.device)


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


def _paired_face_and_wrench_quality(
  slot_quality: torch.Tensor,
  normal_object: torch.Tensor,
  force_object: torch.Tensor,
  force_magnitude: torch.Tensor,
  primary_count: int,
  slots_per_primary: int,
  contacts_for_full_area: int,
  links_for_full_area: int,
  minimum_force: float,
  friction_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compose opposite-face coverage and friction-safe wrench quality."""
  batch_size = slot_quality.shape[0]
  positive_alignment = normal_object.clamp(min=0.0)
  negative_alignment = (-normal_object).clamp(min=0.0)
  contacts_per_side = max(1, math.ceil(contacts_for_full_area / 2))
  links_per_side = max(1, math.ceil(links_for_full_area / 2))
  positive_slot_coverage = (
    (slot_quality[..., None] * positive_alignment).sum(dim=1)
    / float(contacts_per_side)
  ).clamp(max=1.0)
  negative_slot_coverage = (
    (slot_quality[..., None] * negative_alignment).sum(dim=1)
    / float(contacts_per_side)
  ).clamp(max=1.0)
  primary_slot_quality = slot_quality.reshape(
    batch_size, primary_count, slots_per_primary, 1
  )
  primary_positive = (
    primary_slot_quality
    * positive_alignment.reshape(batch_size, primary_count, slots_per_primary, 3)
  ).amax(dim=2)
  primary_negative = (
    primary_slot_quality
    * negative_alignment.reshape(batch_size, primary_count, slots_per_primary, 3)
  ).amax(dim=2)
  positive_link_coverage = (
    (primary_positive > 0.0).float().sum(dim=1) / float(links_per_side)
  ).clamp(max=1.0)
  negative_link_coverage = (
    (primary_negative > 0.0).float().sum(dim=1) / float(links_per_side)
  ).clamp(max=1.0)
  positive_face_quality = positive_slot_coverage * positive_link_coverage
  negative_face_quality = negative_slot_coverage * negative_link_coverage
  paired_face_area = torch.minimum(
    positive_face_quality, negative_face_quality
  ).amax(dim=1)

  normal_force = torch.abs(torch.sum(force_object * normal_object, dim=-1))
  tangential_force = torch.sqrt(
    (torch.square(force_magnitude) - torch.square(normal_force)).clamp(min=0.0)
  )
  friction_capacity = friction_coefficient * normal_force
  friction_margin = (
    (friction_capacity - tangential_force)
    / friction_capacity.clamp_min(1.0e-6)
  ).clamp(min=0.0, max=1.0)
  normal_strength = (
    normal_force / max(minimum_force * 4.0, 1.0e-6)
  ).clamp(max=1.0)
  force_quality = slot_quality * friction_margin * normal_strength
  positive_capacity = (
    force_quality[..., None] * positive_alignment
  ).sum(dim=1) / float(contacts_per_side)
  negative_capacity = (
    force_quality[..., None] * negative_alignment
  ).sum(dim=1) / float(contacts_per_side)
  positive_capacity = positive_capacity.clamp(max=1.0)
  negative_capacity = negative_capacity.clamp(max=1.0)
  capacity_pair = torch.minimum(positive_capacity, negative_capacity)
  force_balance = 1.0 - torch.abs(positive_capacity - negative_capacity) / (
    positive_capacity + negative_capacity
  ).clamp_min(1.0e-6)
  task_wrench_quality = (capacity_pair * force_balance).amax(dim=1)
  grasp_quality = (
    2.0 * paired_face_area * task_wrench_quality
    / (paired_face_area + task_wrench_quality).clamp_min(1.0e-6)
  )
  return paired_face_area, task_wrench_quality, grasp_quality


def _contact_face_metrics(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  object_half_extents: tuple[float, float, float] | None,
  minimum_force: float,
  contacts_for_full_area: int,
  links_for_full_area: int,
  friction_coefficient: float = 1.5,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Estimate paired face coverage, task-wrench quality, and persistence.

  MuJoCo exposes contact points rather than a compliant contact patch area.  For
  a cube, the proxy rewards force-bearing contact slots whose normals align with
  a principal object face and whose points lie away from face edges.  A valid
  face score must cover *both* sides of one principal axis, so pressing several
  links against one face cannot earn a grasp reward.  The task-wrench score also
  requires balanced opposing forces with positive friction-cone safety margin.
  """
  cache_key = (
    env.common_step_counter,
    sensor_name,
    object_half_extents,
    minimum_force,
    contacts_for_full_area,
    links_for_full_area,
    friction_coefficient,
    object_cfg.name,
  )
  cached = getattr(env, "_talos_face_contact_metrics_cache", None)
  if cached is not None and cached[0] == cache_key:
    return cached[1]

  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor):
    raise TypeError(f"'{sensor_name}' must be a ContactSensor.")
  data = sensor.data
  if any(field is None for field in (data.found, data.force, data.pos, data.normal)):
    raise TypeError(
      f"'{sensor_name}' must provide found, force, pos, and normal contact data."
    )

  assert data.found is not None
  assert data.force is not None
  assert data.pos is not None
  assert data.normal is not None
  obj: Entity = env.scene[object_cfg.name]
  batch_size, contact_slots = data.found.shape
  object_quat = obj.data.root_link_quat_w[:, None, :].expand(-1, contact_slots, -1)
  object_pos = obj.data.root_link_pos_w[:, None, :]
  normal_object = quat_apply_inverse(
    object_quat.reshape(-1, 4), data.normal.reshape(-1, 3)
  ).reshape(batch_size, contact_slots, 3)
  contact_pos_object = quat_apply_inverse(
    object_quat.reshape(-1, 4), (data.pos - object_pos).reshape(-1, 3)
  ).reshape(batch_size, contact_slots, 3)

  force_magnitude = torch.linalg.vector_norm(data.force, dim=-1)
  active = (data.found > 0) & (force_magnitude >= minimum_force)
  normal_abs = torch.abs(normal_object)

  if object_half_extents is None:
    face_alignment = torch.ones_like(force_magnitude)
    interior_margin = torch.ones_like(force_magnitude)
  else:
    half_extents = torch.tensor(
      object_half_extents, device=env.device, dtype=contact_pos_object.dtype
    )
    dominant_axis = normal_abs.argmax(dim=-1)
    face_alignment = (
      (normal_abs.amax(dim=-1) - math.sqrt(0.5)) / (1.0 - math.sqrt(0.5))
    ).clamp(min=0.0, max=1.0)
    tangent_mask = 1.0 - torch.nn.functional.one_hot(
      dominant_axis, num_classes=3
    ).to(contact_pos_object.dtype)
    tangent_position = torch.abs(contact_pos_object) / half_extents
    maximum_tangent_position = (tangent_position * tangent_mask).amax(dim=-1)
    interior_margin = (1.0 - maximum_tangent_position).clamp(min=0.0, max=1.0)

  slot_quality = active.float() * face_alignment * interior_margin
  slot_coverage = (
    slot_quality.sum(dim=1) / float(contacts_for_full_area)
  ).clamp(max=1.0)

  slots_per_primary = sensor.cfg.num_slots
  primary_count = len(sensor.primary_names)
  if contact_slots != primary_count * slots_per_primary:
    raise ValueError("Contact slots do not match primary-major sensor layout.")
  primary_quality = slot_quality.reshape(
    batch_size, primary_count, slots_per_primary
  ).amax(dim=-1)
  link_coverage = (
    (primary_quality > 0.0).float().sum(dim=1) / float(links_for_full_area)
  ).clamp(max=1.0)
  area_proxy = slot_coverage * link_coverage

  # Approximate the task-oriented tactile delta metric using measured force
  # margins to the Coulomb friction cone.  The pure helper is unit-tested with
  # one-sided, antipodal, and sliding contact configurations.
  force_object = quat_apply_inverse(
    object_quat.reshape(-1, 4), data.force.reshape(-1, 3)
  ).reshape(batch_size, contact_slots, 3)
  paired_face_area, task_wrench_quality, grasp_quality = (
    _paired_face_and_wrench_quality(
      slot_quality,
      normal_object,
      force_object,
      force_magnitude,
      primary_count,
      slots_per_primary,
      contacts_for_full_area,
      links_for_full_area,
      minimum_force,
      friction_coefficient,
    )
  )

  if data.current_contact_time is None:
    persistence = torch.ones(batch_size, device=env.device)
  else:
    persistent_links = torch.topk(
      data.current_contact_time, k=min(2, data.current_contact_time.shape[1]), dim=1
    ).values
    persistence = persistent_links.amin(dim=1)

  env.extras["log"]["Metrics/face_contact_area_proxy"] = area_proxy.mean()
  env.extras["log"]["Metrics/paired_face_area"] = paired_face_area.mean()
  env.extras["log"]["Metrics/force_closure"] = task_wrench_quality.mean()
  env.extras["log"]["Metrics/task_wrench_quality"] = task_wrench_quality.mean()
  env.extras["log"]["Metrics/verified_grasp_quality"] = grasp_quality.mean()
  env.extras["log"]["Metrics/grasp_contact_persistence_s"] = persistence.mean()
  metrics = (paired_face_area, task_wrench_quality, grasp_quality, persistence)
  env._talos_face_contact_metrics_cache = (cache_key, metrics)
  return metrics


def gripper_face_contact_area(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  object_half_extents: tuple[float, float, float] | None = None,
  minimum_force: float = 0.25,
  contacts_for_full_area: int = 6,
  links_for_full_area: int = 3,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Reward broad force-bearing contacts in the interior of object faces."""
  area, _, _, _ = _contact_face_metrics(
    env,
    sensor_name,
    object_half_extents,
    minimum_force,
    contacts_for_full_area,
    links_for_full_area,
    object_cfg=object_cfg,
  )
  return area


def gripper_force_closure(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  object_half_extents: tuple[float, float, float] | None = None,
  minimum_force: float = 0.25,
  contacts_for_full_area: int = 6,
  links_for_full_area: int = 3,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Reward balanced force-bearing contacts on opposite object faces."""
  _, closure, _, _ = _contact_face_metrics(
    env,
    sensor_name,
    object_half_extents,
    minimum_force,
    contacts_for_full_area,
    links_for_full_area,
    object_cfg=object_cfg,
  )
  return closure


def stable_face_grasp_success(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  object_half_extents: tuple[float, float, float] | None = None,
  minimum_force: float = 0.25,
  contacts_for_full_area: int = 6,
  links_for_full_area: int = 3,
  minimum_area: float = 0.30,
  minimum_force_closure: float = 0.25,
  minimum_duration_s: float = 0.15,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Sparse success for a persistent, broad, force-closure grasp."""
  area, closure, _, persistence = _contact_face_metrics(
    env,
    sensor_name,
    object_half_extents,
    minimum_force,
    contacts_for_full_area,
    links_for_full_area,
    object_cfg=object_cfg,
  )
  success = (
    (area >= minimum_area)
    & (closure >= minimum_force_closure)
    & (persistence >= minimum_duration_s)
  )
  env.extras["log"]["Metrics/face_grasp_success"] = success.float().mean()
  return success.float()


class grasp_quality_record_progress:
  """Reward only record improvements in verified grasp quality.

  Unlike additive contact rewards, this term cannot be farmed by maintaining a
  static touch.  Paired face placement and task-wrench quality must improve in
  the same episode for additional reward to be produced.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._best = torch.zeros(env.num_envs, device=env.device)
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    object_half_extents: tuple[float, float, float] | None = None,
    minimum_force: float = 0.25,
    contacts_for_full_area: int = 6,
    links_for_full_area: int = 3,
    friction_coefficient: float = 1.5,
    object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  ) -> torch.Tensor:
    _, _, quality, _ = _contact_face_metrics(
      env,
      sensor_name,
      object_half_extents,
      minimum_force,
      contacts_for_full_area,
      links_for_full_area,
      friction_coefficient,
      object_cfg,
    )
    improvement = torch.where(
      self._initialized,
      (quality - self._best).clamp(min=0.0),
      torch.zeros_like(quality),
    )
    self._best = torch.where(
      self._initialized, torch.maximum(self._best, quality), quality
    )
    self._initialized[:] = True
    env.extras["log"]["Metrics/grasp_quality_record"] = self._best.mean()
    env.extras["log"]["Metrics/grasp_quality_record_improvement"] = (
      improvement.mean()
    )
    return improvement / env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._best[env_ids] = 0.0
    self._initialized[env_ids] = False


class sustained_grasp_ready_success:
  """Emit one success pulse after holding a physical grasp-quality threshold."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._required_steps = max(
      1, math.ceil(cfg.params["required_duration_s"] / env.step_dt)
    )
    self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._emitted = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    required_duration_s: float,
    minimum_grasp_quality: float,
    object_half_extents: tuple[float, float, float] | None = None,
    minimum_force: float = 0.25,
    contacts_for_full_area: int = 6,
    links_for_full_area: int = 3,
    friction_coefficient: float = 1.5,
    object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  ) -> torch.Tensor:
    del required_duration_s
    _, _, quality, _ = _contact_face_metrics(
      env,
      sensor_name,
      object_half_extents,
      minimum_force,
      contacts_for_full_area,
      links_for_full_area,
      friction_coefficient,
      object_cfg,
    )
    valid = quality >= minimum_grasp_quality
    self._counter = torch.where(valid, self._counter + 1, 0)
    success = self._counter >= self._required_steps
    pulse = success & ~self._emitted
    self._emitted |= success
    env.extras["log"]["Metrics/grasp_ready_valid"] = valid.float().mean()
    env.extras["log"]["Metrics/grasp_ready_success"] = success.float().mean()
    return pulse.float() / env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._counter[env_ids] = 0
    self._emitted[env_ids] = False


def _object_lift_state(
  obj: Entity,
  initial_center_height: float,
  table_height: float,
  object_half_extents: tuple[float, float, float] | None,
  sphere_radius: float | None,
  clearance_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  center_height = obj.data.root_link_pos_w[:, 2]
  center_lift = center_height - initial_center_height
  if object_half_extents is not None:
    half_extents = torch.tensor(
      object_half_extents, device=center_height.device, dtype=center_height.dtype
    )
    rotation = matrix_from_quat(obj.data.root_link_quat_w)
    vertical_radius = (torch.abs(rotation[:, 2, :]) * half_extents).sum(dim=1)
  elif sphere_radius is not None:
    vertical_radius = torch.full_like(center_height, sphere_radius)
  else:
    raise ValueError("Specify object_half_extents or sphere_radius.")
  bottom_clearance = (
    center_height - vertical_radius - table_height - clearance_margin
  ).clamp(min=0.0)
  effective_lift = torch.minimum(center_lift.clamp(min=0.0), bottom_clearance)
  return center_lift, bottom_clearance, effective_lift


def object_lift_progress(
  env: ManagerBasedRlEnv,
  initial_center_height: float,
  table_height: float,
  target_lift_height: float,
  sensor_name: str,
  object_half_extents: tuple[float, float, float] | None = None,
  sphere_radius: float | None = None,
  exponent: float = 2.0,
  clearance_margin: float = 0.002,
  minimum_force: float = 0.25,
  contacts_for_full_area: int = 6,
  links_for_full_area: int = 3,
  minimum_grasp_duration_s: float = 0.10,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Reward stable-grasp lift with quadratic, true-airborne height progress."""
  obj: Entity = env.scene[object_cfg.name]
  center_lift, clearance, effective_lift = _object_lift_state(
    obj,
    initial_center_height,
    table_height,
    object_half_extents,
    sphere_radius,
    clearance_margin,
  )
  progress = (effective_lift / target_lift_height).clamp(max=1.0).pow(exponent)
  area, closure, _, persistence = _contact_face_metrics(
    env,
    sensor_name,
    object_half_extents,
    minimum_force,
    contacts_for_full_area,
    links_for_full_area,
    object_cfg=object_cfg,
  )
  duration_gate = (persistence / minimum_grasp_duration_s).clamp(max=1.0)
  grasp_gate = torch.sqrt((area * closure).clamp(min=0.0)) * duration_gate
  env.extras["log"]["Metrics/object_com_lift_height_m"] = center_lift.mean()
  env.extras["log"]["Metrics/object_lift_height_m"] = center_lift.mean()
  env.extras["log"]["Metrics/object_bottom_clearance_m"] = clearance.mean()
  env.extras["log"]["Metrics/object_effective_lift_m"] = effective_lift.mean()
  env.extras["log"]["Metrics/stable_grasp_gate"] = grasp_gate.mean()
  return progress * grasp_gate


class object_lift_record_progress:
  """Reward new, grasp-verified airborne-height records only."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._best = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    initial_center_height: float,
    table_height: float,
    target_lift_height: float,
    sensor_name: str,
    object_half_extents: tuple[float, float, float] | None = None,
    sphere_radius: float | None = None,
    clearance_margin: float = 0.002,
    minimum_grasp_quality: float = 0.10,
    minimum_force: float = 0.25,
    contacts_for_full_area: int = 6,
    links_for_full_area: int = 3,
    friction_coefficient: float = 1.5,
    object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_cfg.name]
    center_lift, clearance, effective_lift = _object_lift_state(
      obj,
      initial_center_height,
      table_height,
      object_half_extents,
      sphere_radius,
      clearance_margin,
    )
    progress = (effective_lift / target_lift_height).clamp(max=1.0)
    _, _, grasp_quality, _ = _contact_face_metrics(
      env,
      sensor_name,
      object_half_extents,
      minimum_force,
      contacts_for_full_area,
      links_for_full_area,
      friction_coefficient,
      object_cfg,
    )
    quality_gate = (grasp_quality / minimum_grasp_quality).clamp(0.0, 1.0)
    verified_progress = progress * quality_gate
    improvement = (verified_progress - self._best).clamp(min=0.0)
    self._best = torch.maximum(self._best, verified_progress)
    env.extras["log"]["Metrics/object_com_lift_height_m"] = center_lift.mean()
    env.extras["log"]["Metrics/object_bottom_clearance_m"] = clearance.mean()
    env.extras["log"]["Metrics/verified_lift_progress"] = verified_progress.mean()
    env.extras["log"]["Metrics/lift_progress_record"] = self._best.mean()
    return improvement / env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._best[env_ids] = 0.0


class sustained_verified_pick_success:
  """One-shot pick success after a grasped object is lifted and held.

  Success requires paired-face task-wrench quality, true bottom clearance, low
  object speed, and low hand-object relative speed for a continuous window.
  This rejects touch-only policies, transient ballistic launches, and brittle
  grasps which lose the object during the hold test.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._required_steps = max(
      1, math.ceil(cfg.params["required_duration_s"] / env.step_dt)
    )
    self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._emitted = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self._previous_relative_position = torch.zeros(
      (env.num_envs, 3), device=env.device
    )
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    site_name: str,
    initial_center_height: float,
    table_height: float,
    minimum_lift_height: float,
    required_duration_s: float,
    minimum_grasp_quality: float,
    maximum_relative_speed: float,
    maximum_object_speed: float,
    metric_prefix: str,
    object_half_extents: tuple[float, float, float] | None = None,
    sphere_radius: float | None = None,
    clearance_margin: float = 0.002,
    minimum_force: float = 0.25,
    contacts_for_full_area: int = 6,
    links_for_full_area: int = 3,
    friction_coefficient: float = 1.5,
    robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  ) -> torch.Tensor:
    del required_duration_s
    robot: Entity = env.scene[robot_cfg.name]
    obj: Entity = env.scene[object_cfg.name]
    _, _, effective_lift = _object_lift_state(
      obj,
      initial_center_height,
      table_height,
      object_half_extents,
      sphere_radius,
      clearance_margin,
    )
    _, _, grasp_quality, _ = _contact_face_metrics(
      env,
      sensor_name,
      object_half_extents,
      minimum_force,
      contacts_for_full_area,
      links_for_full_area,
      friction_coefficient,
      object_cfg,
    )
    site_id = robot.site_names.index(site_name)
    relative_position = obj.data.root_link_pos_w - robot.data.site_pos_w[:, site_id]
    relative_speed = torch.linalg.vector_norm(
      relative_position - self._previous_relative_position, dim=1
    ) / env.step_dt
    relative_speed = torch.where(
      self._initialized, relative_speed, torch.zeros_like(relative_speed)
    )
    self._previous_relative_position.copy_(relative_position)
    self._initialized[:] = True
    object_speed = torch.linalg.vector_norm(
      obj.data.root_link_vel_w[:, :3], dim=1
    )
    valid = (
      (effective_lift >= minimum_lift_height)
      & (grasp_quality >= minimum_grasp_quality)
      & (relative_speed <= maximum_relative_speed)
      & (object_speed <= maximum_object_speed)
    )
    self._counter = torch.where(valid, self._counter + 1, 0)
    success = self._counter >= self._required_steps
    pulse = success & ~self._emitted
    self._emitted |= success
    env.extras["log"][f"Metrics/{metric_prefix}_valid"] = valid.float().mean()
    env.extras["log"][f"Metrics/{metric_prefix}_success"] = success.float().mean()
    env.extras["log"]["Metrics/object_hand_relative_speed_mps"] = (
      relative_speed.mean()
    )
    env.extras["log"]["Metrics/object_linear_speed_mps"] = object_speed.mean()
    return pulse.float() / env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._counter[env_ids] = 0
    self._emitted[env_ids] = False
    self._previous_relative_position[env_ids] = 0.0
    self._initialized[env_ids] = False


def gripper_reopening_during_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize opening the coupled gripper while it is touching the object."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
    raise TypeError(f"'{sensor_name}' must provide ContactSensor found data.")
  robot: Entity = env.scene[asset_cfg.name]
  touching = (sensor.data.found > 0).any(dim=1)
  reopening_speed = robot.data.joint_vel[:, asset_cfg.joint_ids].clamp(min=0.0)
  penalty = torch.square(reopening_speed).sum(dim=1) * touching.float()
  env.extras["log"]["Metrics/gripper_reopening_contact"] = penalty.mean()
  return penalty


def object_launch_velocity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  initial_center_height: float,
  maximum_upward_speed: float,
  minimum_center_lift: float = 0.005,
  object_half_extents: tuple[float, float, float] | None = None,
  minimum_force: float = 0.25,
  contacts_for_full_area: int = 6,
  links_for_full_area: int = 3,
  friction_coefficient: float = 1.5,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Penalize upward ballistic motion not supported by a verified grasp."""
  obj: Entity = env.scene[object_cfg.name]
  _, _, grasp_quality, _ = _contact_face_metrics(
    env,
    sensor_name,
    object_half_extents,
    minimum_force,
    contacts_for_full_area,
    links_for_full_area,
    friction_coefficient,
    object_cfg,
  )
  center_lift = obj.data.root_link_pos_w[:, 2] - initial_center_height
  upward_excess = (
    obj.data.root_link_vel_w[:, 2] - maximum_upward_speed
  ).clamp(min=0.0)
  airborne = center_lift >= minimum_center_lift
  penalty = (
    torch.square(upward_excess / maximum_upward_speed)
    * (1.0 - grasp_quality)
    * airborne.float()
  )
  env.extras["log"]["Metrics/object_launch_penalty"] = penalty.mean()
  return penalty


def excessive_object_lift_height(
  env: ManagerBasedRlEnv,
  initial_center_height: float,
  maximum_center_lift_height: float = 0.30,
  excess_scale: float = 0.10,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Quadratic hinge cost when object COM rises more than the safe limit."""
  obj: Entity = env.scene[object_cfg.name]
  center_lift = obj.data.root_link_pos_w[:, 2] - initial_center_height
  excess = (center_lift - maximum_center_lift_height).clamp(min=0.0)
  env.extras["log"]["Metrics/object_excessive_height_m"] = excess.mean()
  return torch.square(excess / excess_scale).clamp(max=4.0)


def grasp_and_lift_success(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  initial_center_height: float,
  table_height: float,
  minimum_lift_height: float,
  minimum_contacts: int = 2,
  object_half_extents: tuple[float, float, float] | None = None,
  sphere_radius: float | None = None,
  clearance_margin: float = 0.002,
  minimum_face_area: float = 0.30,
  minimum_force_closure: float = 0.25,
  minimum_force: float = 0.25,
  contacts_for_full_area: int = 6,
  links_for_full_area: int = 3,
  minimum_grasp_duration_s: float = 0.10,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Reward the transition state needed by the later transport policy."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
    raise TypeError(f"'{sensor_name}' must provide ContactSensor found data.")
  obj: Entity = env.scene[object_cfg.name]
  enough_contacts = sensor.data.found.sum(dim=1) >= minimum_contacts
  _, _, effective_lift = _object_lift_state(
    obj,
    initial_center_height,
    table_height,
    object_half_extents,
    sphere_radius,
    clearance_margin,
  )
  area, closure, _, persistence = _contact_face_metrics(
    env,
    sensor_name,
    object_half_extents,
    minimum_force,
    contacts_for_full_area,
    links_for_full_area,
    object_cfg=object_cfg,
  )
  high_enough = effective_lift >= minimum_lift_height
  stable_grasp = (
    (area >= minimum_face_area)
    & (closure >= minimum_force_closure)
    & (persistence >= minimum_grasp_duration_s)
  )
  success = enough_contacts & high_enough & stable_grasp
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
  return _per_foot_contact(env, sensor_name).all(dim=1).float()


def _per_foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Return one contact-presence column per configured foot."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
    raise TypeError(f"'{sensor_name}' must provide ContactSensor found data.")
  found = sensor.data.found > 0
  num_slots = sensor.cfg.num_slots
  per_foot = found.reshape(env.num_envs, len(sensor.primary_names), num_slots).any(
    dim=2
  )
  return per_foot


def _target_xy_w(
  env: ManagerBasedRlEnv,
  target_position: tuple[float, float],
) -> torch.Tensor:
  target_xy = torch.tensor(
    target_position, device=env.device, dtype=torch.float
  ).repeat(env.num_envs, 1)
  return target_xy + env.scene.env_origins[:, :2]


def _heading_error_to_target(
  env: ManagerBasedRlEnv,
  asset: Entity,
  target_position: tuple[float, float],
) -> torch.Tensor:
  """Return signed yaw error from the robot forward axis to a world target."""
  target_xy = _target_xy_w(env, target_position)
  offset_w = target_xy - asset.data.root_link_pos_w[:, :2]
  offset_w_3d = torch.cat(
    (offset_w, torch.zeros((env.num_envs, 1), device=env.device)), dim=1
  )
  offset_b = quat_apply_inverse(asset.data.root_link_quat_w, offset_w_3d)
  return torch.atan2(offset_b[:, 1], offset_b[:, 0])


def _foot_slip_speeds(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]
  contact = _per_foot_contact(env, sensor_name)
  foot_velocity_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  if foot_velocity_xy.shape[1] != contact.shape[1]:
    raise ValueError("Foot sites and contact primaries must have matching counts.")
  slip_speed = torch.linalg.vector_norm(foot_velocity_xy, dim=2)
  return slip_speed, contact


def feet_gait_contact_tracking(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_position: tuple[float, float],
  period: float,
  offsets: tuple[float, float],
  stance_ratio: float,
  stop_distance: float,
) -> torch.Tensor:
  """Reward alternating foot contacts according to a periodic biped gait."""
  if period <= 0.0:
    raise ValueError("gait period must be positive")
  if len(offsets) != 2:
    raise ValueError("biped gait requires exactly two phase offsets")
  if not 0.0 < stance_ratio < 1.0:
    raise ValueError("stance_ratio must be in (0, 1)")

  contact = _per_foot_contact(env, sensor_name)
  if contact.shape[1] != len(offsets):
    raise ValueError("gait offsets must match the configured feet")
  global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
  phase_offsets = torch.tensor(offsets, device=env.device, dtype=torch.float)
  expected_stance = ((global_phase + phase_offsets) % 1.0) < stance_ratio
  match_fraction = (expected_stance == contact).float().mean(dim=1)

  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  target_xy = _target_xy_w(env, target_position)
  distance = torch.linalg.vector_norm(
    asset.data.root_link_pos_w[:, :2] - target_xy, dim=1
  )
  active = distance > stop_distance
  env.extras["log"]["Metrics/gait_contact_match"] = match_fraction.mean()
  return match_fraction * active.float()


def contact_foot_slip_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize horizontal velocity of feet that are carrying ground contact."""
  slip_speed, contact = _foot_slip_speeds(env, sensor_name, asset_cfg)
  cost = torch.sum(torch.square(slip_speed) * contact.float(), dim=1)
  count = contact.float().sum().clamp_min(1.0)
  mean_slip = torch.sum(slip_speed * contact.float()) / count
  env.extras["log"]["Metrics/foot_slip_speed_mps"] = mean_slip
  return cost


class feet_swing_peak_height:
  """Reward swing-foot peak height when a genuine airborne step lands."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    num_feet = len(asset_cfg.site_ids)
    self._peak_height = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float
    )
    self._was_airborne = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.bool
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_position: tuple[float, float],
    target_height: float,
    std: float,
    minimum_air_time: float,
    stop_distance: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    if target_height <= 0.0 or std <= 0.0:
      raise ValueError("target_height and std must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, ContactSensor):
      raise TypeError(f"'{sensor_name}' must be a ContactSensor.")
    if sensor.data.last_air_time is None:
      raise RuntimeError("Foot contact sensor must enable track_air_time.")

    contact = _per_foot_contact(env, sensor_name)
    foot_height = (
      asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
      - env.scene.env_origins[:, 2].unsqueeze(1)
    )
    airborne = ~contact
    self._peak_height = torch.where(
      airborne,
      torch.maximum(self._peak_height, foot_height),
      self._peak_height,
    )
    self._was_airborne |= airborne
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    valid_landing = (
      first_contact
      & self._was_airborne
      & (sensor.data.last_air_time >= minimum_air_time)
    )
    landing_reward = torch.exp(
      -torch.square((self._peak_height - target_height) / std)
    ) * valid_landing.float()

    target_xy = _target_xy_w(env, target_position)
    distance = torch.linalg.vector_norm(
      asset.data.root_link_pos_w[:, :2] - target_xy, dim=1
    )
    reward = landing_reward.sum(dim=1) * (distance > stop_distance).float()
    landing_count = valid_landing.float().sum().clamp_min(1.0)
    env.extras["log"]["Metrics/swing_peak_height_m"] = (
      (self._peak_height * valid_landing.float()).sum() / landing_count
    )
    self._peak_height = torch.where(
      first_contact, torch.zeros_like(self._peak_height), self._peak_height
    )
    self._was_airborne = torch.where(
      first_contact, torch.zeros_like(self._was_airborne), self._was_airborne
    )
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._peak_height[env_ids] = 0.0
    self._was_airborne[env_ids] = False


def base_facing_target_exp(
  env: ManagerBasedRlEnv,
  target_position: tuple[float, float],
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the robot for pointing its forward axis at the table."""
  if std <= 0.0:
    raise ValueError("heading std must be positive")
  asset: Entity = env.scene[asset_cfg.name]
  error = _heading_error_to_target(env, asset, target_position)
  env.extras["log"]["Metrics/table_heading_error_rad"] = torch.abs(error).mean()
  return torch.exp(-torch.square(error / std))


def base_lateral_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize sideways base velocity in the robot frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 1])


def both_feet_contact_near_target(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_position: tuple[float, float],
  distance_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a double-support stop only after reaching the target area."""
  asset: Entity = env.scene[asset_cfg.name]
  target_xy = _target_xy_w(env, target_position)
  close = torch.linalg.vector_norm(
    asset.data.root_link_pos_w[:, :2] - target_xy, dim=1
  ) <= distance_threshold
  return both_feet_contact(env, sensor_name) * close.float()


def _stable_standing_mask(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  maximum_projected_gravity_xy: float,
  maximum_linear_speed: float,
  maximum_angular_speed: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  feet = both_feet_contact(env, sensor_name) > 0.5
  tilt = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
  lin_speed = torch.linalg.vector_norm(asset.data.root_link_vel_w[:, :3], dim=1)
  ang_speed = torch.linalg.vector_norm(asset.data.root_link_vel_w[:, 3:], dim=1)
  return (
    feet
    & (tilt <= maximum_projected_gravity_xy)
    & (lin_speed <= maximum_linear_speed)
    & (ang_speed <= maximum_angular_speed)
  )


class sustained_standing_success:
  """Sparse success after continuously satisfying a stable standing condition."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._required_steps = max(
      1, math.ceil(cfg.params["required_duration_s"] / env.step_dt)
    )
    self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    required_duration_s: float,
    maximum_projected_gravity_xy: float,
    maximum_linear_speed: float,
    maximum_angular_speed: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del required_duration_s
    stable = _stable_standing_mask(
      env,
      sensor_name,
      maximum_projected_gravity_xy,
      maximum_linear_speed,
      maximum_angular_speed,
      asset_cfg,
    )
    self._counter = torch.where(stable, self._counter + 1, 0)
    success = self._counter >= self._required_steps
    env.extras["log"]["Metrics/standing_success"] = success.float().mean()
    return success.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._counter[env_ids] = 0


def base_target_distance_tanh(
  env: ManagerBasedRlEnv,
  target_position: tuple[float, float],
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense navigation reward for moving the base to an environment-local target."""
  asset: Entity = env.scene[asset_cfg.name]
  target_xy = torch.tensor(
    target_position, device=env.device, dtype=torch.float
  ).repeat(env.num_envs, 1)
  target_xy += env.scene.env_origins[:, :2]
  distance = torch.linalg.vector_norm(
    asset.data.root_link_pos_w[:, :2] - target_xy, dim=1
  )
  env.extras["log"]["Metrics/base_target_distance_m"] = distance.mean()
  return 1.0 - torch.tanh(distance / std)


def _base_target_radial_velocity(
  asset: Entity,
  target_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return planar speed and signed world-frame velocity toward a target."""
  target_offset = target_xy - asset.data.root_link_pos_w[:, :2]
  target_direction = target_offset / torch.linalg.vector_norm(
    target_offset, dim=1, keepdim=True
  ).clamp_min(1.0e-6)
  planar_velocity = asset.data.root_link_vel_w[:, :2]
  planar_speed = torch.linalg.vector_norm(planar_velocity, dim=1)
  radial_velocity = torch.sum(planar_velocity * target_direction, dim=1)
  return planar_speed, radial_velocity


class base_target_progress:
  """Reward measured base velocity toward the planar navigation target.

  The reward manager integrates this velocity with ``env.step_dt``, so the
  episode return is proportional to net target-directed displacement.  Reading
  MuJoCo's root velocity directly also makes zero physical speed produce exactly
  zero progress reward.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg, env

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_position: tuple[float, float],
    maximum_speed: float,
    maximum_projected_gravity_xy: float,
    heading_target_position: tuple[float, float] | None = None,
    sensor_name: str | None = None,
    minimum_height: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    target_xy = _target_xy_w(env, target_position)
    planar_speed, radial_velocity = _base_target_radial_velocity(asset, target_xy)
    radial_velocity = radial_velocity.clamp(-maximum_speed, maximum_speed)
    tilt = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
    upright_scale = (1.0 - tilt / maximum_projected_gravity_xy).clamp(0.0, 1.0)
    supported = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if sensor_name is not None:
      supported = _per_foot_contact(env, sensor_name).any(dim=1)
    valid_posture = supported & (asset.data.root_link_pos_w[:, 2] >= minimum_height)
    heading_gate = torch.ones(env.num_envs, device=env.device)
    if heading_target_position is not None:
      heading_error = _heading_error_to_target(env, asset, heading_target_position)
      heading_gate = torch.cos(heading_error).clamp(0.0, 1.0)
    env.extras["log"]["Metrics/base_lin_vel_x_mps"] = (
      asset.data.root_link_vel_w[:, 0].mean()
    )
    env.extras["log"]["Metrics/base_lin_vel_y_mps"] = (
      asset.data.root_link_vel_w[:, 1].mean()
    )
    env.extras["log"]["Metrics/base_planar_speed_mps"] = planar_speed.mean()
    env.extras["log"]["Metrics/base_target_radial_speed_mps"] = (
      radial_velocity.mean()
    )
    env.extras["log"]["Metrics/navigation_reward_gate"] = (
      valid_posture.float().mean()
    )
    env.extras["log"]["Metrics/navigation_heading_gate"] = heading_gate.mean()
    return radial_velocity * heading_gate * upright_scale * valid_posture.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids


def base_target_speed_above_threshold(
  env: ManagerBasedRlEnv,
  target_position: tuple[float, float],
  minimum_speed: float,
  target_speed: float,
  maximum_projected_gravity_xy: float,
  heading_target_position: tuple[float, float] | None = None,
  sensor_name: str | None = None,
  minimum_height: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward deliberate target-directed motion, with no reward below a deadband."""
  if not 0.0 <= minimum_speed < target_speed:
    raise ValueError("Expected 0 <= minimum_speed < target_speed.")
  asset: Entity = env.scene[asset_cfg.name]
  target_xy = _target_xy_w(env, target_position)
  _, radial_velocity = _base_target_radial_velocity(asset, target_xy)
  speed_scale = (
    (radial_velocity - minimum_speed) / (target_speed - minimum_speed)
  ).clamp(0.0, 1.0)
  tilt = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
  upright_scale = (1.0 - tilt / maximum_projected_gravity_xy).clamp(0.0, 1.0)
  supported = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  if sensor_name is not None:
    supported = _per_foot_contact(env, sensor_name).any(dim=1)
  valid_posture = supported & (asset.data.root_link_pos_w[:, 2] >= minimum_height)
  heading_gate = torch.ones(env.num_envs, device=env.device)
  if heading_target_position is not None:
    heading_error = _heading_error_to_target(env, asset, heading_target_position)
    heading_gate = torch.cos(heading_error).clamp(0.0, 1.0)
  return speed_scale * heading_gate * upright_scale * valid_posture.float()


class sustained_healthy_gait_success:
  """Sparse success after both feet complete phase-aligned healthy steps."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._required_steps = max(
      1, math.ceil(cfg.params["required_duration_s"] / env.step_dt)
    )
    sensor = env.scene[cfg.params["sensor_name"]]
    if not isinstance(sensor, ContactSensor):
      raise TypeError("Healthy gait success requires a foot ContactSensor.")
    self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._completed_steps = torch.zeros(
      (env.num_envs, len(sensor.primary_names)),
      dtype=torch.bool,
      device=env.device,
    )
    self._peak_height = torch.zeros_like(self._completed_steps, dtype=torch.float)
    self._was_airborne = torch.zeros_like(self._completed_steps)
    self._gait_match_ema = torch.zeros(
      env.num_envs, dtype=torch.float, device=env.device
    )
    gait_match_window_s = cfg.params["gait_match_window_s"]
    if gait_match_window_s <= 0.0:
      raise ValueError("gait_match_window_s must be positive")
    self._gait_match_alpha = min(1.0, env.step_dt / gait_match_window_s)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_position: tuple[float, float],
    heading_target_position: tuple[float, float],
    required_duration_s: float,
    minimum_radial_speed: float,
    maximum_radial_speed: float,
    maximum_projected_gravity_xy: float,
    maximum_heading_error: float,
    maximum_contact_slip_speed: float,
    minimum_height: float,
    minimum_step_air_time: float,
    minimum_swing_peak_height: float,
    maximum_swing_peak_height: float,
    gait_period: float,
    gait_offsets: tuple[float, float],
    gait_stance_ratio: float,
    gait_match_window_s: float,
    minimum_gait_contact_match: float,
    foot_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del required_duration_s, gait_match_window_s
    if gait_period <= 0.0:
      raise ValueError("gait_period must be positive")
    if len(gait_offsets) != self._completed_steps.shape[1]:
      raise ValueError("gait_offsets must match the configured feet")
    if not 0.0 < gait_stance_ratio < 1.0:
      raise ValueError("gait_stance_ratio must be in (0, 1)")
    if not 0.0 <= minimum_gait_contact_match <= 1.0:
      raise ValueError("minimum_gait_contact_match must be in [0, 1]")
    if not 0.0 < minimum_swing_peak_height < maximum_swing_peak_height:
      raise ValueError("swing peak height bounds must be positive and ordered")
    asset: Entity = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, ContactSensor) or sensor.data.last_air_time is None:
      raise RuntimeError("Healthy gait success requires tracked foot air time.")

    contact = _per_foot_contact(env, sensor_name)
    foot_height = (
      asset.data.site_pos_w[:, foot_asset_cfg.site_ids, 2]
      - env.scene.env_origins[:, 2].unsqueeze(1)
    )
    airborne = ~contact
    self._peak_height = torch.where(
      airborne,
      torch.maximum(self._peak_height, foot_height),
      self._peak_height,
    )
    self._was_airborne |= airborne
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    valid_landing = first_contact & self._was_airborne & (
      sensor.data.last_air_time >= minimum_step_air_time
    )
    healthy_landing = (
      valid_landing
      & (self._peak_height >= minimum_swing_peak_height)
      & (self._peak_height <= maximum_swing_peak_height)
    )
    self._completed_steps |= healthy_landing
    self._peak_height = torch.where(
      first_contact, torch.zeros_like(self._peak_height), self._peak_height
    )
    self._was_airborne = torch.where(
      first_contact, torch.zeros_like(self._was_airborne), self._was_airborne
    )

    global_phase = (
      (env.episode_length_buf * env.step_dt) % gait_period / gait_period
    ).unsqueeze(1)
    phase_offsets = torch.tensor(
      gait_offsets, device=env.device, dtype=torch.float
    )
    expected_stance = (
      (global_phase + phase_offsets) % 1.0
    ) < gait_stance_ratio
    gait_match = (expected_stance == contact).float().mean(dim=1)
    self._gait_match_ema.lerp_(gait_match, self._gait_match_alpha)

    target_xy = _target_xy_w(env, target_position)
    _, radial_velocity = _base_target_radial_velocity(asset, target_xy)
    tilt = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
    heading_error = torch.abs(
      _heading_error_to_target(env, asset, heading_target_position)
    )
    slip_speed, _ = _foot_slip_speeds(env, sensor_name, foot_asset_cfg)
    contact_slip = torch.where(contact, slip_speed, torch.zeros_like(slip_speed))
    low_slip = contact_slip.amax(dim=1) <= maximum_contact_slip_speed
    supported = contact.any(dim=1)
    healthy = (
      self._completed_steps.all(dim=1)
      & (self._gait_match_ema >= minimum_gait_contact_match)
      & supported
      & (radial_velocity >= minimum_radial_speed)
      & (radial_velocity <= maximum_radial_speed)
      & (tilt <= maximum_projected_gravity_xy)
      & (heading_error <= maximum_heading_error)
      & low_slip
      & (asset.data.root_link_pos_w[:, 2] >= minimum_height)
    )
    self._counter = torch.where(healthy, self._counter + 1, 0)
    success = self._counter >= self._required_steps
    env.extras["log"]["Metrics/healthy_gait_success"] = success.float().mean()
    env.extras["log"]["Metrics/healthy_gait_steps_completed"] = (
      self._completed_steps.float().mean()
    )
    env.extras["log"]["Metrics/healthy_gait_contact_match_ema"] = (
      self._gait_match_ema.mean()
    )
    return success.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._counter[env_ids] = 0
    self._completed_steps[env_ids] = False
    self._peak_height[env_ids] = 0.0
    self._was_airborne[env_ids] = False
    self._gait_match_ema[env_ids] = 0.0


class sustained_navigation_success:
  """Sparse success for arriving at the table while remaining balanced."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._required_steps = max(
      1, math.ceil(cfg.params["required_duration_s"] / env.step_dt)
    )
    self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    sensor = env.scene[cfg.params["sensor_name"]]
    if not isinstance(sensor, ContactSensor):
      raise TypeError("Navigation success requires a foot ContactSensor.")
    self._completed_steps = torch.zeros(
      (env.num_envs, len(sensor.primary_names)),
      dtype=torch.bool,
      device=env.device,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_position: tuple[float, float],
    distance_threshold: float,
    required_duration_s: float,
    maximum_projected_gravity_xy: float,
    maximum_linear_speed: float,
    maximum_angular_speed: float,
    heading_target_position: tuple[float, float],
    maximum_heading_error: float,
    maximum_contact_slip_speed: float,
    minimum_step_air_time: float,
    foot_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del required_duration_s
    asset: Entity = env.scene[asset_cfg.name]
    target_xy = _target_xy_w(env, target_position)
    close = (
      torch.linalg.vector_norm(asset.data.root_link_pos_w[:, :2] - target_xy, dim=1)
      <= distance_threshold
    )
    stable = _stable_standing_mask(
      env,
      sensor_name,
      maximum_projected_gravity_xy,
      maximum_linear_speed,
      maximum_angular_speed,
      asset_cfg,
    )
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, ContactSensor) or sensor.data.last_air_time is None:
      raise RuntimeError("Navigation success requires tracked foot air time.")
    valid_landing = sensor.compute_first_contact(dt=env.step_dt) & (
      sensor.data.last_air_time >= minimum_step_air_time
    )
    self._completed_steps |= valid_landing
    heading_error = torch.abs(
      _heading_error_to_target(env, asset, heading_target_position)
    )
    slip_speed, contact = _foot_slip_speeds(env, sensor_name, foot_asset_cfg)
    contact_slip = torch.where(contact, slip_speed, torch.zeros_like(slip_speed))
    low_slip = contact_slip.amax(dim=1) <= maximum_contact_slip_speed
    healthy_arrival = (
      close
      & stable
      & (heading_error <= maximum_heading_error)
      & low_slip
      & self._completed_steps.all(dim=1)
    )
    self._counter = torch.where(healthy_arrival, self._counter + 1, 0)
    success = self._counter >= self._required_steps
    env.extras["log"]["Metrics/navigation_success"] = success.float().mean()
    return success.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._counter[env_ids] = 0
    self._completed_steps[env_ids] = False


def object_target_distance_tanh(
  env: ManagerBasedRlEnv,
  target_position: tuple[float, float, float],
  std: float,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Dense transport reward from the object center to its placement target."""
  obj: Entity = env.scene[object_cfg.name]
  target_w = torch.tensor(target_position, device=env.device, dtype=torch.float).repeat(
    env.num_envs, 1
  )
  target_w += env.scene.env_origins
  distance = torch.linalg.vector_norm(obj.data.root_link_pos_w - target_w, dim=1)
  env.extras["log"]["Metrics/object_target_distance_m"] = distance.mean()
  return 1.0 - torch.tanh(distance / std)


class place_object_success:
  """Reward lifting, transporting, and releasing the object in the target zone."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    initial_center_height: float,
    minimum_lift_height: float,
    target_position: tuple[float, float, float],
    target_xy_tolerance: float,
    target_height_tolerance: float,
    maximum_release_contacts: int = 0,
    object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
      raise TypeError(f"'{sensor_name}' must provide ContactSensor found data.")
    obj: Entity = env.scene[object_cfg.name]
    self._lifted |= (
      obj.data.root_link_pos_w[:, 2] - initial_center_height
    ) >= minimum_lift_height

    target_w = torch.tensor(
      target_position, device=env.device, dtype=torch.float
    ).repeat(env.num_envs, 1)
    target_w += env.scene.env_origins
    target_xy_error = torch.linalg.vector_norm(
      obj.data.root_link_pos_w[:, :2] - target_w[:, :2], dim=1
    )
    target_height_error = torch.abs(obj.data.root_link_pos_w[:, 2] - target_w[:, 2])
    contacts = sensor.data.found.sum(dim=1)
    success = (
      self._lifted
      & (target_xy_error <= target_xy_tolerance)
      & (target_height_error <= target_height_tolerance)
      & (contacts <= maximum_release_contacts)
    )
    env.extras["log"]["Metrics/place_success"] = success.float().mean()
    return success.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._lifted[env_ids] = False


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
