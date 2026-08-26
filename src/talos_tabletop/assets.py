"""Procedural tabletop assets used by the manipulation tasks."""

from functools import partial
from typing import Literal

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ObjectShape = Literal["cube", "sphere"]

# The table is deliberately a floating fixed cuboid without legs. Its front
# edge is 0.40 m in front of the robot origin, leaving knee clearance while the
# 0.80 m top surface stays slightly below the nominal 0.893 m grasp centers.
TABLE_TOP_HEIGHT_M = 0.80
TABLE_TOP_HALF_SIZE_M = (0.35, 0.60, 0.04)
TABLE_CENTER_X_M = 0.75

# The blue object starts on the green pickup zone. The red zone is the target
# used by the later transport/place policy; both zones are visual-only sites.
PICKUP_ZONE_CENTER_M = (0.58, -0.18)
TARGET_ZONE_CENTER_M = (0.62, 0.22)
ZONE_HALF_SIZE_M = (0.10, 0.10, 0.001)

CUBE_HALF_SIZE_M = (0.03, 0.03, 0.03)
SPHERE_RADIUS_M = 0.035
OBJECT_MASS_KG = 0.25


def object_initial_position(object_shape: ObjectShape) -> tuple[float, float, float]:
  """Return the object-center position on the pickup zone."""
  support_radius = (
    CUBE_HALF_SIZE_M[2] if object_shape == "cube" else SPHERE_RADIUS_M
  )
  return (
    PICKUP_ZONE_CENTER_M[0],
    PICKUP_ZONE_CENTER_M[1],
    TABLE_TOP_HEIGHT_M + support_radius,
  )


def get_table_spec() -> mujoco.MjSpec:
  """Create a fixed collision tabletop with visual pickup and target zones."""
  spec = mujoco.MjSpec()
  table = spec.worldbody.add_body(name="table")
  table.add_geom(
    name="table_top_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_TOP_HALF_SIZE_M,
    rgba=(0.42, 0.42, 0.45, 1.0),
    friction=(1.0, 0.02, 0.001),
  )

  zone_z = TABLE_TOP_HALF_SIZE_M[2] + ZONE_HALF_SIZE_M[2] + 1.0e-4
  table.add_site(
    name="pickup_zone",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(
      PICKUP_ZONE_CENTER_M[0] - TABLE_CENTER_X_M,
      PICKUP_ZONE_CENTER_M[1],
      zone_z,
    ),
    size=ZONE_HALF_SIZE_M,
    rgba=(0.10, 0.72, 0.20, 0.75),
  )
  table.add_site(
    name="target_zone",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(
      TARGET_ZONE_CENTER_M[0] - TABLE_CENTER_X_M,
      TARGET_ZONE_CENTER_M[1],
      zone_z,
    ),
    size=ZONE_HALF_SIZE_M,
    rgba=(0.90, 0.08, 0.08, 0.75),
  )
  return spec


def get_object_spec(object_shape: ObjectShape = "cube") -> mujoco.MjSpec:
  """Create a blue free rigid cube or sphere for grasp training."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_freejoint")

  if object_shape == "cube":
    geom_type = mujoco.mjtGeom.mjGEOM_BOX
    geom_size = CUBE_HALF_SIZE_M
  elif object_shape == "sphere":
    geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom_size = (SPHERE_RADIUS_M, 0.0, 0.0)
  else:
    raise ValueError(f"Unsupported object shape: {object_shape!r}")

  body.add_geom(
    name="object_collision",
    type=geom_type,
    size=geom_size,
    mass=OBJECT_MASS_KG,
    rgba=(0.05, 0.28, 0.92, 1.0),
    friction=(1.0, 0.02, 0.001),
  )
  body.add_site(name="object_center", size=(0.008,), rgba=(0.0, 0.0, 1.0, 0.0))
  return spec


def get_table_cfg() -> EntityCfg:
  """Return the fixed floating-tabletop entity configuration."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(TABLE_CENTER_X_M, 0.0, TABLE_TOP_HEIGHT_M - TABLE_TOP_HALF_SIZE_M[2]),
      joint_pos={},
    ),
    collisions=(
      CollisionCfg(
        geom_names_expr=("table_top_collision",),
        condim=4,
        friction=(1.0, 0.02, 0.001),
      ),
    ),
    spec_fn=get_table_spec,
  )


def get_object_cfg(object_shape: ObjectShape = "cube") -> EntityCfg:
  """Return a free blue object initialized on the green pickup zone."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=object_initial_position(object_shape),
      joint_pos={},
    ),
    collisions=(
      CollisionCfg(
        geom_names_expr=("object_collision",),
        condim=4,
        priority=2,
        friction=(1.0, 0.02, 0.001),
      ),
    ),
    spec_fn=partial(get_object_spec, object_shape=object_shape),
  )
