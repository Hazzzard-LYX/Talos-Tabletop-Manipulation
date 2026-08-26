import mujoco
import pytest
from mjlab.entity import Entity
from talos_tabletop.assets import (
  CUBE_HALF_SIZE_M,
  OBJECT_MASS_KG,
  SPHERE_RADIUS_M,
  TABLE_TOP_HALF_SIZE_M,
  get_object_spec,
  get_table_spec,
)
from talos_tabletop.robots import (
  TALOS_GRIPPER_MAIN_JOINT_NAMES,
  get_talos_grasping_robot_cfg,
)


def test_migrated_talos_grippers_compile_with_real_coupling() -> None:
  entity = Entity(get_talos_grasping_robot_cfg())
  model = entity.spec.compile()

  assert model.nu == 32
  assert model.neq == 12
  assert tuple(model.actuator(i).name for i in range(model.nu - 2, model.nu)) == (
    *TALOS_GRIPPER_MAIN_JOINT_NAMES,
  )
  assert model.geom("head_1_collision").type == mujoco.mjtGeom.mjGEOM_MESH
  assert model.geom("head_2_collision").type == mujoco.mjtGeom.mjGEOM_MESH


def test_table_is_fixed_and_has_visual_task_zones() -> None:
  table = get_table_spec().compile()

  assert table.body("table").jntnum == 0
  assert tuple(table.geom("table_top_collision").size) == TABLE_TOP_HALF_SIZE_M
  assert table.site("pickup_zone").id >= 0
  assert table.site("target_zone").id >= 0
  # Zones are sites rather than geoms, so they cannot create contacts.
  assert table.ngeom == 1


@pytest.mark.parametrize(
  ("shape", "geom_type", "first_size"),
  (
    ("cube", mujoco.mjtGeom.mjGEOM_BOX, CUBE_HALF_SIZE_M[0]),
    ("sphere", mujoco.mjtGeom.mjGEOM_SPHERE, SPHERE_RADIUS_M),
  ),
)
def test_rigid_objects_are_free_and_blue(
  shape: str,
  geom_type: mujoco.mjtGeom,
  first_size: float,
) -> None:
  obj = get_object_spec(shape).compile()

  assert obj.body("object").jntnum == 1
  assert obj.joint("object_freejoint").type == mujoco.mjtJoint.mjJNT_FREE
  assert obj.body("object").mass[0] == pytest.approx(OBJECT_MASS_KG)
  assert obj.geom("object_collision").type == geom_type
  assert obj.geom("object_collision").size[0] == pytest.approx(first_size)
  assert tuple(obj.geom("object_collision").rgba[:3]) == pytest.approx(
    (0.05, 0.28, 0.92)
  )
