"""PAL Robotics Talos constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from talos_tabletop import TALOS_TABLETOP_SRC_PATH

##
# MJCF and assets.
##

TALOS_XML: Path = TALOS_TABLETOP_SRC_PATH / "robots" / "talos" / "xmls" / "talos.xml"
assert TALOS_XML.exists()
TALOS_GRIPPER_ASSET_DIR = TALOS_XML.parent / "assets" / "gripper"

TALOS_PAYLOAD_BODY_NAME = "front_payload"
TALOS_PAYLOAD_GEOM_NAME = "front_payload_collision"
TALOS_PAYLOAD_PARENT_BODY_NAME = "torso_2_link"
TALOS_PAYLOAD_MASS = 10.0
TALOS_PAYLOAD_HALF_SIZE = (0.12, 0.12, 0.12)
TALOS_PAYLOAD_POS = (0.35, 0.0, 0.14)

TALOS_TRAY_BODY_NAME = "hand_tray"
TALOS_TRAY_PARENT_BODY_NAME = "arm_left_7_link"
TALOS_TRAY_SECONDARY_BODY_NAME = "arm_right_7_link"
TALOS_TRAY_RIGHT_MOUNT_SITE_NAME = "hand_tray_right_mount"
TALOS_TRAY_RIGHT_WRIST_SITE_NAME = "right_wrist_tray_mount"
TALOS_TRAY_WELD_NAME = "right_hand_tray_weld"
TALOS_TRAY_MASS = 3.0
TALOS_TRAY_HALF_SIZE = (0.275, 0.39, 0.0125)
TALOS_TRAY_PAYLOAD_BODY_NAME = "tray_payload"
TALOS_TRAY_PAYLOAD_JOINT_NAME = "tray_payload_freejoint"
TALOS_TRAY_PAYLOAD_GEOM_NAME = "tray_payload_collision"
TALOS_TRAY_PAYLOAD_MASS = 2.5
TALOS_TRAY_PAYLOAD_HALF_SIZE = (0.12, 0.12, 0.12)
TALOS_TRAY_PAYLOAD_INIT_POS = (0.43, 0.0, 1.2275)
TALOS_FREE_TRAY_INIT_POS = (0.43, 0.0, 1.09)
TALOS_FREE_TRAY_JOINT_NAME = "hand_tray_freejoint"

# Provisional rigid-contact model for badminton overgrip tape.  This is an
# intentionally exposed calibration parameter, not a claim that every tape has
# the same coefficient.  Training randomizes around it in the grasping task.
TALOS_OVERGRIP_FRICTION = (1.5, 0.02, 0.001)

TALOS_GRIPPER_MAIN_JOINT_NAMES = (
  "gripper_left_joint",
  "gripper_right_joint",
)
TALOS_GRIPPER_FINGERTIP_BODY_NAMES = (
  "gripper_left_fingertip_1_link",
  "gripper_left_fingertip_2_link",
  "gripper_left_fingertip_3_link",
  "gripper_right_fingertip_1_link",
  "gripper_right_fingertip_2_link",
  "gripper_right_fingertip_3_link",
)
TALOS_GRIPPER_CONTACT_BODY_NAMES = tuple(
  f"gripper_{side}_{link}_link"
  for side in ("left", "right")
  for link in (
    "motor_double",
    "inner_double",
    "fingertip_1",
    "fingertip_2",
    "motor_single",
    "inner_single",
    "fingertip_3",
  )
)

# TALOS exposes four six-axis force/torque sensors: one at each wrist and
# ankle.  The MuJoCo sites are placed on the child side of each physical
# transducer so the builtin force and torque sensors report the load
# transmitted through that joint.  Keep this ordering stable: it is part of
# the deployable actor observation contract.
TALOS_FT_SITE_BODIES = {
  "left_wrist_ft": "arm_left_7_link",
  "right_wrist_ft": "arm_right_7_link",
  "left_ankle_ft": "leg_left_6_link",
  "right_ankle_ft": "leg_right_6_link",
}
TALOS_WRIST_FT_SITE_NAMES = ("left_wrist_ft", "right_wrist_ft")
TALOS_ANKLE_FT_SITE_NAMES = ("left_ankle_ft", "right_ankle_ft")
TALOS_FT_SITE_NAMES = TALOS_WRIST_FT_SITE_NAMES + TALOS_ANKLE_FT_SITE_NAMES

# Joint-level torque feedback is available on TALOS except at the head and
# wrists (the grippers are fixed in this model).  These expressions select
# torso, arm 1--5, and all leg joints: 24 measurements in model order.
TALOS_TORQUE_SENSOR_JOINT_NAMES = (
  r"torso_.*_joint",
  r"arm_.*_[1-5]_joint",
  r"leg_.*_joint",
)

# The tray mounting transform is calibrated against INIT_STATE.  At that pose,
# the tray is horizontal at world position (0.43, 0.0, 1.09), while the two
# wrist origins are 0.723 m apart.  Two straight cylindrical handles extend
# vertically down from the tray into the gripper grasp centers.  A site weld
# closes the kinematic chain at the right grasp center so that both arms carry
# the tray load.
TALOS_TRAY_POS_LEFT_WRIST = (
  0.31250841649220595,
  -0.23132470677882747,
  -0.22389631487034964,
)
TALOS_TRAY_QUAT_LEFT_WRIST = (
  0.809667734759874,
  -0.05229616376117401,
  0.5489815516628436,
  0.20080469735176915,
)
TALOS_TRAY_RIGHT_MOUNT_POS = (
  -0.046019897496159,
  -0.332312499153264,
  -0.196777633603233,
)
TALOS_TRAY_GRASP_POS_WRIST = (0.0, 0.0, -0.213985)
TALOS_TRAY_RIGHT_WRIST_SITE_QUAT = (
  0.809667734759874,
  0.05229616376117401,
  0.5489815516628436,
  -0.20080469735176915,
)
TALOS_TRAY_HANDLE_POSITIONS = {
  "left": (-0.046019897496159, 0.332312499153264, -0.1225),
  "right": (-0.046019897496159, -0.332312499153264, -0.1225),
}

TALOS_TRAY_GRIPPER_BODY_NAMES = (
  TALOS_TRAY_PARENT_BODY_NAME,
  "gripper_left_motor_double_link",
  "gripper_left_inner_double_link",
  "gripper_left_fingertip_1_link",
  "gripper_left_fingertip_2_link",
  "gripper_left_motor_single_link",
  "gripper_left_inner_single_link",
  "gripper_left_fingertip_3_link",
  TALOS_TRAY_SECONDARY_BODY_NAME,
  "gripper_right_motor_double_link",
  "gripper_right_inner_double_link",
  "gripper_right_fingertip_1_link",
  "gripper_right_fingertip_2_link",
  "gripper_right_motor_single_link",
  "gripper_right_inner_single_link",
  "gripper_right_fingertip_3_link",
)


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TALOS_XML))
  for site_name, body_name in TALOS_FT_SITE_BODIES.items():
    body = spec.body(body_name)
    if body is None:
      raise ValueError(
        f"TALOS F/T sensor body '{body_name}' was not found in the model."
      )
    body.add_site(name=site_name, size=(0.008,))
  return spec


def _add_gripper_articulation(spec: mujoco.MjSpec) -> None:
  """Restore the coupled TALOS three-finger grippers for contact tasks.

  The source MJCF keeps these joints and collision meshes disabled so legacy
  locomotion checkpoints retain their original dimensions.  This task-local
  augmentation follows the PAL URDF transmission: one actuated joint per hand
  and six mimic joints (three linkage joints and three fingertip joints).
  """
  collision_meshes = {
    "gripper_motor_double_collision": "gripper_motor_double_collision.stl",
    "gripper_motor_single_collision": "gripper_motor_single_collision.stl",
    "inner_double_collision": "inner_double_collision.stl",
    "inner_single_collision": "inner_single_collision.stl",
    "fingertip_collision": "fingertip_collision.stl",
  }
  for mesh_name, filename in collision_meshes.items():
    mesh_path = TALOS_GRIPPER_ASSET_DIR / filename
    if not mesh_path.exists():
      raise FileNotFoundError(f"Missing TALOS gripper collision mesh: {mesh_path}")
    spec.add_mesh(name=mesh_name, file=str(mesh_path))

  joint_specs = (
    ("gripper_left_motor_double_link", "gripper_left_joint", (-0.959931, 0.0)),
    (
      "gripper_left_inner_double_link",
      "gripper_left_inner_double_joint",
      (-1.0472, 0.0),
    ),
    (
      "gripper_left_motor_single_link",
      "gripper_left_motor_single_joint",
      (0.0, 1.0472),
    ),
    (
      "gripper_left_inner_single_link",
      "gripper_left_inner_single_joint",
      (0.0, 1.0472),
    ),
    ("gripper_left_fingertip_1_link", "gripper_left_fingertip_1_joint", (0.0, 1.0472)),
    ("gripper_left_fingertip_2_link", "gripper_left_fingertip_2_joint", (0.0, 1.0472)),
    ("gripper_left_fingertip_3_link", "gripper_left_fingertip_3_joint", (0.0, 1.0472)),
    ("gripper_right_motor_double_link", "gripper_right_joint", (-0.959931, 0.0)),
    (
      "gripper_right_inner_double_link",
      "gripper_right_inner_double_joint",
      (-1.0472, 0.0),
    ),
    (
      "gripper_right_motor_single_link",
      "gripper_right_motor_single_joint",
      (0.0, 1.0472),
    ),
    (
      "gripper_right_inner_single_link",
      "gripper_right_inner_single_joint",
      (0.0, 1.0472),
    ),
    (
      "gripper_right_fingertip_1_link",
      "gripper_right_fingertip_1_joint",
      (0.0, 1.0472),
    ),
    (
      "gripper_right_fingertip_2_link",
      "gripper_right_fingertip_2_joint",
      (0.0, 1.0472),
    ),
    (
      "gripper_right_fingertip_3_link",
      "gripper_right_fingertip_3_joint",
      (0.0, 1.0472),
    ),
  )
  for body_name, joint_name, joint_range in joint_specs:
    body = spec.body(body_name)
    if body is None:
      raise ValueError(f"TALOS gripper body '{body_name}' was not found.")
    body.add_joint(
      name=joint_name,
      axis=(1.0, 0.0, 0.0),
      limited=True,
      range=joint_range,
      damping=1.0,
      frictionloss=1.0,
    )

  collision_geoms = (
    (
      "gripper_left_motor_double_link",
      "left_motor_double_grasp_collision",
      "gripper_motor_double_collision",
    ),
    (
      "gripper_left_inner_double_link",
      "left_inner_double_grasp_collision",
      "inner_double_collision",
    ),
    (
      "gripper_left_motor_single_link",
      "left_motor_single_grasp_collision",
      "gripper_motor_single_collision",
    ),
    (
      "gripper_left_inner_single_link",
      "left_inner_single_grasp_collision",
      "inner_single_collision",
    ),
    (
      "gripper_left_fingertip_1_link",
      "left_fingertip_1_grasp_collision",
      "fingertip_collision",
    ),
    (
      "gripper_left_fingertip_2_link",
      "left_fingertip_2_grasp_collision",
      "fingertip_collision",
    ),
    (
      "gripper_left_fingertip_3_link",
      "left_fingertip_3_grasp_collision",
      "fingertip_collision",
    ),
    (
      "gripper_right_motor_double_link",
      "right_motor_double_grasp_collision",
      "gripper_motor_double_collision",
    ),
    (
      "gripper_right_inner_double_link",
      "right_inner_double_grasp_collision",
      "inner_double_collision",
    ),
    (
      "gripper_right_motor_single_link",
      "right_motor_single_grasp_collision",
      "gripper_motor_single_collision",
    ),
    (
      "gripper_right_inner_single_link",
      "right_inner_single_grasp_collision",
      "inner_single_collision",
    ),
    (
      "gripper_right_fingertip_1_link",
      "right_fingertip_1_grasp_collision",
      "fingertip_collision",
    ),
    (
      "gripper_right_fingertip_2_link",
      "right_fingertip_2_grasp_collision",
      "fingertip_collision",
    ),
    (
      "gripper_right_fingertip_3_link",
      "right_fingertip_3_grasp_collision",
      "fingertip_collision",
    ),
  )
  for body_name, geom_name, mesh_name in collision_geoms:
    spec.body(body_name).add_geom(
      name=geom_name,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      condim=4,
      priority=2,
      friction=TALOS_OVERGRIP_FRICTION,
    )

  # Equality direction and signs reproduce the source model's mechanical
  # linkage.  The initial converted MJCF omitted the fingertip mimic entries;
  # PAL's URDF specifies all three fingertip joints as -1 times the main joint.
  for side in ("left", "right"):
    couplings = (
      (f"gripper_{side}_motor_single_joint", f"gripper_{side}_joint", -1.0),
      (f"gripper_{side}_joint", f"gripper_{side}_inner_double_joint", 1.0),
      (
        f"gripper_{side}_inner_single_joint",
        f"gripper_{side}_inner_double_joint",
        -1.0,
      ),
      (
        f"gripper_{side}_fingertip_1_joint",
        f"gripper_{side}_joint",
        -1.0,
      ),
      (
        f"gripper_{side}_fingertip_2_joint",
        f"gripper_{side}_joint",
        -1.0,
      ),
      (
        f"gripper_{side}_fingertip_3_joint",
        f"gripper_{side}_joint",
        -1.0,
      ),
    )
    for joint1, joint2, ratio in couplings:
      spec.add_equality(
        name=f"{joint1}_coupling",
        type=mujoco.mjtEq.mjEQ_JOINT,
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        name1=joint1,
        name2=joint2,
        data=(0.0, ratio, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        solref=(0.01, 1.0),
      )

  # Adjacent links in each gripper are mechanically connected and should not
  # create self-contact constraints.  Contacts with the tray remain enabled.
  local_excludes = (
    ("arm_{side}_7_link", "gripper_{side}_motor_double_link"),
    ("arm_{side}_7_link", "gripper_{side}_inner_double_link"),
    ("arm_{side}_7_link", "gripper_{side}_motor_single_link"),
    ("arm_{side}_7_link", "gripper_{side}_inner_single_link"),
    ("gripper_{side}_motor_double_link", "gripper_{side}_inner_double_link"),
    ("gripper_{side}_motor_double_link", "gripper_{side}_fingertip_2_link"),
    ("gripper_{side}_motor_double_link", "gripper_{side}_motor_single_link"),
    ("gripper_{side}_motor_double_link", "gripper_{side}_inner_single_link"),
    ("gripper_{side}_inner_double_link", "gripper_{side}_motor_single_link"),
    ("gripper_{side}_fingertip_1_link", "gripper_{side}_fingertip_2_link"),
    ("gripper_{side}_fingertip_1_link", "gripper_{side}_fingertip_3_link"),
    ("gripper_{side}_fingertip_2_link", "gripper_{side}_fingertip_3_link"),
    ("gripper_{side}_motor_single_link", "gripper_{side}_inner_single_link"),
    ("gripper_{side}_motor_single_link", "gripper_{side}_fingertip_3_link"),
  )
  for side in ("left", "right"):
    for index, (body1, body2) in enumerate(local_excludes):
      spec.add_exclude(
        name=f"gripper_{side}_self_{index}",
        bodyname1=body1.format(side=side),
        bodyname2=body2.format(side=side),
      )


def get_grasping_spec() -> mujoco.MjSpec:
  """Return TALOS with articulated, collision-enabled coupled grippers."""
  spec = get_spec()
  _add_gripper_articulation(spec)
  for side in ("left", "right"):
    spec.body(f"arm_{side}_7_link").add_site(
      name=f"{side}_grasp_center",
      pos=TALOS_TRAY_GRASP_POS_WRIST,
      size=(0.008,),
    )
  return spec


def get_payload_spec() -> mujoco.MjSpec:
  """Return TALOS with a rigid cube payload fixed in front of its torso.

  MuJoCo represents a fixed connection by making a body a child of another body
  without adding a joint.  The payload therefore contributes mass and inertia to
  the articulated system but introduces no additional degree of freedom.
  """
  spec = get_spec()
  torso = spec.body(TALOS_PAYLOAD_PARENT_BODY_NAME)
  if torso is None:
    raise ValueError(
      f"TALOS body '{TALOS_PAYLOAD_PARENT_BODY_NAME}' was not found in the model."
    )

  payload = torso.add_body(
    name=TALOS_PAYLOAD_BODY_NAME,
    pos=TALOS_PAYLOAD_POS,
  )
  payload.add_geom(
    name=TALOS_PAYLOAD_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TALOS_PAYLOAD_HALF_SIZE,
    mass=TALOS_PAYLOAD_MASS,
    rgba=(0.85, 0.15, 0.05, 1.0),
    friction=(0.8, 0.02, 0.001),
  )
  return spec


def _add_tray_geometry(tray: mujoco.MjsBody) -> None:
  """Add the common rigid tray geometry to ``tray``."""
  tray.add_geom(
    name="hand_tray_base_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TALOS_TRAY_HALF_SIZE,
    mass=1.8,
    rgba=(0.12, 0.32, 0.62, 1.0),
    friction=(1.0, 0.02, 0.001),
  )

  rim_specs = (
    ("front", (0.275, 0.0, 0.0375), (0.0125, 0.39, 0.0375)),
    ("back", (-0.275, 0.0, 0.0375), (0.0125, 0.39, 0.0375)),
    ("left", (0.0, 0.39, 0.0375), (0.275, 0.0125, 0.0375)),
    ("right", (0.0, -0.39, 0.0375), (0.275, 0.0125, 0.0375)),
  )
  for side, pos, size in rim_specs:
    tray.add_geom(
      name=f"hand_tray_{side}_rim_collision",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
      mass=0.15,
      rgba=(0.08, 0.22, 0.50, 1.0),
      friction=(1.0, 0.02, 0.001),
    )

  for side, handle_pos in TALOS_TRAY_HANDLE_POSITIONS.items():
    tray.add_geom(
      name=f"hand_tray_{side}_handle_collision",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      pos=handle_pos,
      size=(0.025, 0.11),
      mass=0.3,
      rgba=(0.12, 0.12, 0.14, 1.0),
      condim=4,
      priority=2,
      friction=TALOS_OVERGRIP_FRICTION,
    )

  for side, handle_pos in TALOS_TRAY_HANDLE_POSITIONS.items():
    grasp_y = handle_pos[1]
    tray.add_site(
      name=f"hand_tray_{side}_handle_grasp",
      pos=(TALOS_TRAY_RIGHT_MOUNT_POS[0], grasp_y, TALOS_TRAY_RIGHT_MOUNT_POS[2]),
      size=(0.008,),
    )


def get_tray_spec() -> mujoco.MjSpec:
  """Return TALOS rigidly carrying an empty tray with both wrists.

  MuJoCo bodies can only have one structural parent.  The tray is therefore a
  fixed child of the left wrist, while a site-based weld connects it to the
  right wrist.  This creates a closed kinematic chain and distributes tray
  forces through both arms without introducing another degree of freedom.
  """
  spec = get_spec()
  left_wrist = spec.body(TALOS_TRAY_PARENT_BODY_NAME)
  right_wrist = spec.body(TALOS_TRAY_SECONDARY_BODY_NAME)
  if left_wrist is None or right_wrist is None:
    raise ValueError("TALOS wrist bodies required for the hand tray were not found.")

  tray = left_wrist.add_body(
    name=TALOS_TRAY_BODY_NAME,
    pos=TALOS_TRAY_POS_LEFT_WRIST,
    quat=TALOS_TRAY_QUAT_LEFT_WRIST,
  )
  _add_tray_geometry(tray)

  tray.add_site(
    name=TALOS_TRAY_RIGHT_MOUNT_SITE_NAME,
    pos=TALOS_TRAY_RIGHT_MOUNT_POS,
  )
  right_wrist.add_site(
    name=TALOS_TRAY_RIGHT_WRIST_SITE_NAME,
    pos=TALOS_TRAY_GRASP_POS_WRIST,
    quat=TALOS_TRAY_RIGHT_WRIST_SITE_QUAT,
  )
  spec.add_equality(
    name=TALOS_TRAY_WELD_NAME,
    type=mujoco.mjtEq.mjEQ_WELD,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    name1=TALOS_TRAY_RIGHT_MOUNT_SITE_NAME,
    name2=TALOS_TRAY_RIGHT_WRIST_SITE_NAME,
    solref=(0.005, 1.0),
  )

  # The current TALOS gripper joints are fixed in the source MJCF.  The handles
  # are aligned visually with their grasp centers, while rigid mounting carries
  # the forces.  Filter redundant hand-handle contacts so they do not fight the
  # closed-chain constraint.
  for gripper_body_name in TALOS_TRAY_GRIPPER_BODY_NAMES:
    spec.add_exclude(
      name=f"tray_{gripper_body_name}_contact",
      bodyname1=TALOS_TRAY_BODY_NAME,
      bodyname2=gripper_body_name,
    )
  return spec


def get_free_hand_tray_spec() -> mujoco.MjSpec:
  """Return the hand tray as an independent six-DoF rigid entity."""
  spec = mujoco.MjSpec()
  tray = spec.worldbody.add_body(name=TALOS_TRAY_BODY_NAME)
  tray.add_freejoint(name=TALOS_FREE_TRAY_JOINT_NAME)
  _add_tray_geometry(tray)
  return spec


def get_free_tray_payload_spec() -> mujoco.MjSpec:
  """Return a standalone cube with a six-DoF freejoint for tray transport."""
  spec = mujoco.MjSpec()
  payload = spec.worldbody.add_body(name=TALOS_TRAY_PAYLOAD_BODY_NAME)
  payload.add_freejoint(name=TALOS_TRAY_PAYLOAD_JOINT_NAME)
  payload.add_geom(
    name=TALOS_TRAY_PAYLOAD_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TALOS_TRAY_PAYLOAD_HALF_SIZE,
    mass=TALOS_TRAY_PAYLOAD_MASS,
    rgba=(0.85, 0.15, 0.05, 1.0),
    friction=(1.0, 0.02, 0.001),
  )
  return spec


##
# Actuator parameters calculs.
##

# params (BeyondMimic paper methodology)
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0  # over-damped
REDUCTION_RATIO = 100
factor = 0.01

# ---- arm joints params

# joints inertia
ARM_1_MOTOR_INERTIA = 0.000207288
ARM_2_MOTOR_INERTIA = 0.000140493
ARM_34_MOTOR_INERTIA = 8.60398e-05
ARM_567_MOTOR_INERTIA = 1.0e-05  # this one has not been properly identified
# joints armature (reflected inertia)
ARM_1_ARMATURE = factor * ARM_1_MOTOR_INERTIA * REDUCTION_RATIO**2
ARM_2_ARMATURE = factor * ARM_2_MOTOR_INERTIA * REDUCTION_RATIO**2
ARM_34_ARMATURE = factor * ARM_34_MOTOR_INERTIA * REDUCTION_RATIO**2
ARM_567_ARMATURE = factor * ARM_567_MOTOR_INERTIA * REDUCTION_RATIO**2
# joints effort limit
ARM_1_EFFORT_LIMIT = 100.0
ARM_2_EFFORT_LIMIT = 100.0
ARM_34_EFFORT_LIMIT = 70.0
ARM_567_EFFORT_LIMIT = 8.0
# joints stiffness
ARM_1_STIFFNESS = ARM_1_ARMATURE * NATURAL_FREQ**2
ARM_2_STIFFNESS = ARM_2_ARMATURE * NATURAL_FREQ**2
ARM_34_STIFFNESS = ARM_34_ARMATURE * NATURAL_FREQ**2
ARM_567_STIFFNESS = ARM_567_ARMATURE * NATURAL_FREQ**2
# joints damping
ARM_1_DAMPING = 2.0 * DAMPING_RATIO * ARM_1_ARMATURE * NATURAL_FREQ
ARM_2_DAMPING = 2.0 * DAMPING_RATIO * ARM_2_ARMATURE * NATURAL_FREQ
ARM_34_DAMPING = 2.0 * DAMPING_RATIO * ARM_34_ARMATURE * NATURAL_FREQ
ARM_567_DAMPING = 2.0 * DAMPING_RATIO * ARM_567_ARMATURE * NATURAL_FREQ

# ---- torso joints params

# joints inertia
TORSO_MOTOR_INERTIA = 0.000207288
# joints armature (reflected inertia)
TORSO_ARMATURE = factor * TORSO_MOTOR_INERTIA * REDUCTION_RATIO**2
# joints effort limit
TORSO_EFFORT_LIMIT = 200.0
# joints stiffness
TORSO_STIFFNESS = TORSO_ARMATURE * NATURAL_FREQ**2
# joints damping
TORSO_DAMPING = 2.0 * DAMPING_RATIO * TORSO_ARMATURE * NATURAL_FREQ

# ---- head joints params

# joints inertia
HEAD_MOTOR_INERTIA = 1.0e-05  # this one has not been properly identified
# joints reduction ratio
HEAD_REDUCTION_RATIO = 144
# joints armature (reflected inertia)
HEAD_ARMATURE = HEAD_MOTOR_INERTIA * HEAD_REDUCTION_RATIO**2
# joints effort limit
HEAD_1_EFFORT_LIMIT = 8.0
HEAD_2_EFFORT_LIMIT = 4.0
# joints stiffness
HEAD_STIFFNESS = HEAD_ARMATURE * NATURAL_FREQ**2
# joints damping limit
HEAD_DAMPING = 2.0 * DAMPING_RATIO * HEAD_ARMATURE * NATURAL_FREQ

# ---- leg joints params

# joints inertia
LEG_16_MOTOR_INERTIA = 8.60398e-05
LEG_235_MOTOR_INERTIA = 0.000207288
LEG_4_MOTOR_INERTIA = 0.000195461
# joints reduction ratio
LEG_1_REDUCTION_RATIO = 150
LEG_26_REDUCTION_RATIO = 101
LEG_4_REDUCTION_RATIO = 144
# joints armature (reflected inertia)
LEG_1_ARMATURE = factor * LEG_16_MOTOR_INERTIA * LEG_1_REDUCTION_RATIO**2
LEG_2_ARMATURE = factor * LEG_235_MOTOR_INERTIA * LEG_26_REDUCTION_RATIO**2
LEG_35_ARMATURE = factor * LEG_235_MOTOR_INERTIA * REDUCTION_RATIO**2
LEG_4_ARMATURE = factor * LEG_4_MOTOR_INERTIA * LEG_4_REDUCTION_RATIO**2
LEG_6_ARMATURE = factor * LEG_16_MOTOR_INERTIA * LEG_26_REDUCTION_RATIO**2
# joints effort limit
LEG_1_EFFORT_LIMIT = 100.0
LEG_2_EFFORT_LIMIT = 160.0
LEG_35_EFFORT_LIMIT = 160.0
LEG_4_EFFORT_LIMIT = 400.0
LEG_6_EFFORT_LIMIT = 100.0
# joints stiffness
LEG_1_STIFFNESS = LEG_1_ARMATURE * NATURAL_FREQ**2
LEG_2_STIFFNESS = LEG_2_ARMATURE * NATURAL_FREQ**2
LEG_35_STIFFNESS = LEG_35_ARMATURE * NATURAL_FREQ**2
LEG_4_STIFFNESS = LEG_4_ARMATURE * NATURAL_FREQ**2
LEG_6_STIFFNESS = LEG_6_ARMATURE * NATURAL_FREQ**2
# joints damping limit
LEG_1_DAMPING = 2.0 * DAMPING_RATIO * LEG_1_ARMATURE * NATURAL_FREQ
LEG_2_DAMPING = 2.0 * DAMPING_RATIO * LEG_2_ARMATURE * NATURAL_FREQ
LEG_35_DAMPING = 2.0 * DAMPING_RATIO * LEG_35_ARMATURE * NATURAL_FREQ
LEG_4_DAMPING = 2.0 * DAMPING_RATIO * LEG_4_ARMATURE * NATURAL_FREQ
LEG_6_DAMPING = 2.0 * DAMPING_RATIO * LEG_6_ARMATURE * NATURAL_FREQ

##
# Actuator config.
##

# arm actuators
ARM_1_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("arm_.*_1_joint",),
  effort_limit=ARM_1_EFFORT_LIMIT,
  armature=ARM_1_ARMATURE,
  stiffness=ARM_1_STIFFNESS,
  damping=ARM_1_DAMPING,
)
ARM_2_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("arm_.*_2_joint",),
  effort_limit=ARM_2_EFFORT_LIMIT,
  armature=ARM_2_ARMATURE,
  stiffness=ARM_2_STIFFNESS,
  damping=ARM_2_DAMPING,
)
ARM_34_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "arm_.*_3_joint",
    "arm_.*_4_joint",
  ),
  effort_limit=ARM_34_EFFORT_LIMIT,
  armature=ARM_34_ARMATURE,
  stiffness=ARM_34_STIFFNESS,
  damping=ARM_34_DAMPING,
)
ARM_567_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "arm_.*_5_joint",
    "arm_.*_6_joint",
    "arm_.*_7_joint",
  ),
  effort_limit=ARM_567_EFFORT_LIMIT,
  armature=ARM_567_ARMATURE,
  stiffness=ARM_567_STIFFNESS,
  damping=ARM_567_DAMPING,
)
# torso actuators
TORSO_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("torso_.*_joint",),
  effort_limit=TORSO_EFFORT_LIMIT,
  armature=TORSO_ARMATURE,
  stiffness=TORSO_STIFFNESS,
  damping=TORSO_DAMPING,
)
# head actuators
HEAD_1_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("head_1_joint",),
  effort_limit=HEAD_1_EFFORT_LIMIT,
  armature=HEAD_ARMATURE,
  stiffness=HEAD_STIFFNESS,
  damping=HEAD_DAMPING,
)
HEAD_2_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("head_2_joint",),
  effort_limit=HEAD_2_EFFORT_LIMIT,
  armature=HEAD_ARMATURE,
  stiffness=HEAD_STIFFNESS,
  damping=HEAD_DAMPING,
)
# leg actuators
LEG_1_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("leg_.*_1_joint",),
  effort_limit=LEG_1_EFFORT_LIMIT,
  armature=LEG_1_ARMATURE,
  stiffness=LEG_1_STIFFNESS,
  damping=LEG_1_DAMPING,
)
LEG_2_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("leg_.*_2_joint",),
  effort_limit=LEG_2_EFFORT_LIMIT,
  armature=LEG_2_ARMATURE,
  stiffness=LEG_2_STIFFNESS,
  damping=LEG_2_DAMPING,
)
LEG_35_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "leg_.*_3_joint",
    "leg_.*_5_joint",
  ),
  effort_limit=LEG_35_EFFORT_LIMIT,
  armature=LEG_35_ARMATURE,
  stiffness=LEG_35_STIFFNESS,
  damping=LEG_35_DAMPING,
)
LEG_4_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("leg_.*_4_joint",),
  effort_limit=LEG_4_EFFORT_LIMIT,
  armature=LEG_4_ARMATURE,
  stiffness=LEG_4_STIFFNESS,
  damping=LEG_4_DAMPING,
)
LEG_6_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("leg_.*_6_joint",),
  effort_limit=LEG_6_EFFORT_LIMIT,
  armature=LEG_6_ARMATURE,
  stiffness=LEG_6_STIFFNESS,
  damping=LEG_6_DAMPING,
)

# TALOS exposes one commanded gripper degree of freedom per hand.  The
# remaining linkage joints are constrained in ``get_grasping_spec`` and the
# distal fingertip joints are passive.  Parameters match the position actuator
# retained in the project's initial MJCF.
GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=TALOS_GRIPPER_MAIN_JOINT_NAMES,
  effort_limit=10.0,
  stiffness=1000.0,
  damping=0.895504,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.0),
  joint_pos={
    # legs
    "leg_.*_1_joint": 0.0,
    "leg_.*_2_joint": 0.0,
    "leg_.*_3_joint": -0.4,
    "leg_.*_4_joint": 0.8,
    "leg_.*_5_joint": -0.4,
    "leg_.*_6_joint": 0.0,
    # arms
    "arm_left_1_joint": 0.3,
    "arm_right_1_joint": -0.3,
    "arm_left_2_joint": 0.4,
    "arm_right_2_joint": -0.4,
    "arm_left_3_joint": -0.5,
    "arm_right_3_joint": 0.5,
    "arm_.*_4_joint": -1.5,
    "arm_.*_5_joint": 0.0,
    "arm_.*_6_joint": 0.0,
    "arm_.*_7_joint": 0.0,
    # head
    "head_1_joint": 0.0,
    "head_2_joint": 0.0,
    # torso
    "torso_1_joint": 0.0,
    "torso_2_joint": 0.15,
  },
  joint_vel={".*": 0.0},
)

GRASPING_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=INIT_STATE.pos,
  rot=INIT_STATE.rot,
  lin_vel=INIT_STATE.lin_vel,
  ang_vel=INIT_STATE.ang_vel,
  joint_pos={
    **(INIT_STATE.joint_pos or {}),
    # Start around a handle instead of asking exploration to discover a grasp
    # from a fully open hand.  The coupling entries make the keyframe satisfy
    # the equality constraints at reset.
    "gripper_.*_inner_double_joint": -0.24,
    "gripper_.*_motor_single_joint": 0.24,
    "gripper_.*_inner_single_joint": 0.24,
    "gripper_.*_fingertip_.*_joint": 0.24,
    "gripper_(left|right)_joint": -0.24,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = ".*_foot_collision"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
)

GRASPING_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    _foot_regex: 3,
    ".*_grasp_collision": 4,
    ".*_collision": 1,
  },
  priority={
    _foot_regex: 1,
    ".*_grasp_collision": 2,
  },
  friction={
    _foot_regex: (0.6,),
    ".*_grasp_collision": TALOS_OVERGRIP_FRICTION,
  },
)
##
# Final config.
##

TALOS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ARM_1_ACTUATOR_CFG,
    ARM_2_ACTUATOR_CFG,
    ARM_34_ACTUATOR_CFG,
    ARM_567_ACTUATOR_CFG,
    LEG_1_ACTUATOR_CFG,
    LEG_2_ACTUATOR_CFG,
    LEG_35_ACTUATOR_CFG,
    LEG_4_ACTUATOR_CFG,
    LEG_6_ACTUATOR_CFG,
    HEAD_1_ACTUATOR_CFG,
    HEAD_2_ACTUATOR_CFG,
    TORSO_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)

TALOS_GRASPING_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(*TALOS_ARTICULATION.actuators, GRIPPER_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)


def get_talos_robot_cfg() -> EntityCfg:
  """Get a fresh Talos robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=TALOS_ARTICULATION,
  )


def get_talos_payload_robot_cfg() -> EntityCfg:
  """Get a fresh TALOS configuration with the rigid front payload."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_payload_spec,
    articulation=TALOS_ARTICULATION,
  )


def get_talos_tray_robot_cfg() -> EntityCfg:
  """Get a fresh TALOS configuration carrying the empty hand tray."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_tray_spec,
    articulation=TALOS_ARTICULATION,
  )


def get_talos_grasping_robot_cfg() -> EntityCfg:
  """Get TALOS with coupled, actuated grippers and restored contact meshes."""
  return EntityCfg(
    init_state=GRASPING_INIT_STATE,
    collisions=(GRASPING_COLLISION,),
    spec_fn=get_grasping_spec,
    articulation=TALOS_GRASPING_ARTICULATION,
  )


def get_talos_free_hand_tray_cfg() -> EntityCfg:
  """Get the independent tray initialized between the two grippers."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=TALOS_FREE_TRAY_INIT_POS,
      joint_pos={},
    ),
    collisions=(
      CollisionCfg(
        geom_names_expr=("hand_tray_.*_collision",),
        condim=4,
        priority=2,
        friction=TALOS_OVERGRIP_FRICTION,
      ),
    ),
    spec_fn=get_free_hand_tray_spec,
  )


def get_talos_free_tray_payload_cfg() -> EntityCfg:
  """Get a standalone free payload initialized at the tray center."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=TALOS_TRAY_PAYLOAD_INIT_POS,
      joint_pos={},
    ),
    spec_fn=get_free_tray_payload_spec,
  )


TALOS_ACTION_SCALE: dict[str, float] = {}

for a in TALOS_ARTICULATION.actuators:
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr

  if not isinstance(e, dict):
    e = {n: e for n in names}
  if not isinstance(s, dict):
    s = {n: s for n in names}

  for n in names:
    if n in e and n in s and s[n]:
      TALOS_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

TALOS_GRASPING_ACTION_SCALE = dict(TALOS_ACTION_SCALE)
# Keep the initially preloaded grasp inside the useful contact range while the
# policy still has high exploration variance.  The generic effort/stiffness
# formula would yield only 0.0025 rad; 0.08 rad still permits active force
# modulation without random first-iteration actions immediately opening the
# hand or driving the linkage deeply through the handle.
for joint_name in TALOS_GRIPPER_MAIN_JOINT_NAMES:
  TALOS_GRASPING_ACTION_SCALE[joint_name] = 0.08


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_talos_robot_cfg())

  viewer.launch(robot.spec.compile())
