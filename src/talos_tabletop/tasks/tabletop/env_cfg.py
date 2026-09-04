"""TALOS tabletop reaching and privileged-position grasping tasks."""

import math
import os
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from talos_tabletop.assets import (
  CUBE_HALF_SIZE_M,
  SPHERE_RADIUS_M,
  TABLE_CENTER_X_M,
  TABLE_TOP_HALF_SIZE_M,
  TABLE_TOP_HEIGHT_M,
  TARGET_ZONE_CENTER_M,
  ObjectShape,
  get_object_cfg,
  get_table_cfg,
  object_initial_position,
)
from talos_tabletop.robots import (
  TALOS_GRASPING_ACTION_SCALE,
  TALOS_GRIPPER_CONTACT_BODY_NAMES,
  get_talos_grasping_robot_cfg,
)
from talos_tabletop.tasks.tabletop import mdp

TABLE_FRONT_X_M = TABLE_CENTER_X_M - TABLE_TOP_HALF_SIZE_M[0]
ROBOT_TABLE_CLEARANCE_M = 3.0
ROBOT_SPAWN_POSITION_M = (
  TABLE_FRONT_X_M - ROBOT_TABLE_CLEARANCE_M,
  0.0,
  1.0,
)
STATIONARY_GRASP_ROBOT_POSITION_M = (0.0, 0.0, 1.0)
STABLE_HOLD_TARGET_LIFT_HEIGHT_M = 0.08
STABLE_HOLD_GRIPPER_POSITION = -0.7306
STABLE_HOLD_OBJECT_POSITION_M = (0.59653, -0.17855, 0.97)
STABLE_HOLD_OBJECT_ROT_WXYZ = (0.993452, 0.0, 0.0, -0.11425)
# IK solution which preserves the selected three-finger cube pose while placing
# the cube center exactly 8 cm above its original tabletop height.
STABLE_HOLD_RIGHT_ARM_JOINT_POS = {
  "torso_1_joint": 0.0795394546,
  "torso_2_joint": 0.0880841381,
  "arm_right_1_joint": 0.9994535929,
  "arm_right_2_joint": -1.4058058187,
  "arm_right_3_joint": 0.4492669156,
  "arm_right_4_joint": -1.1972966978,
  "arm_right_5_joint": 0.0632891793,
  "arm_right_6_joint": 1.0235641200,
  "arm_right_7_joint": -0.0060439499,
}
HIDDEN_SCENE_X_OFFSET_M = 10.0
TABLE_APPROACH_BASE_XY_M = (0.0, 0.0)
MAX_COLLISION_OBSTACLES = 8
LOCOMOTION_PERIOD_S = 0.8
FOOT_SWING_TARGET_HEIGHT_M = 0.09
ControlMode = Literal["position_tracking", "manipulation"]


def make_tabletop_reaching_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the reusable two-hand reaching interface for the tabletop task."""

  ## --------------------------------------------------------
  # Observations
  ## --------------------------------------------------------

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.07, n_max=0.07),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "collision_obstacle_boxes_b": ObservationTermCfg(
      func=mdp.obstacle_boxes_in_robot_frame,
      params={
        "obstacle_names": ("table",),
        "half_extents": (TABLE_TOP_HALF_SIZE_M,),
        "max_obstacles": MAX_COLLISION_OBSTACLES,
      },
    ),
    "pose_command_left": ObservationTermCfg(
      func=mdp.commands_gen,
      params={"command_name": "pose_command_left"},
    ),
    "pose_command_right": ObservationTermCfg(
      func=mdp.commands_gen,
      params={"command_name": "pose_command_right"},
    ),
  }

  critic_terms = {
    **actor_terms,
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ## --------------------------------------------------------
  # Actions
  ## --------------------------------------------------------

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ## --------------------------------------------------------
  # Commands
  ## --------------------------------------------------------

  commands: dict[str, CommandTermCfg] = {
    "pose_command_left": mdp.UniformPoseCommandCfg(
      entity_name="robot",
      debug_vis=True,
      resampling_time_range=(5.0, 10.0),
      site_name="left_grasp_center",
      ranges=mdp.PoseRanges(
        pos_x=(0.0, 0.0),  # Set per-robot.
        pos_y=(0.0, 0.0),  # Set per-robot.
        pos_z=(0.0, 0.0),  # Set per-robot.
      ),
    ),
    "pose_command_right": mdp.UniformPoseCommandCfg(
      entity_name="robot",
      debug_vis=True,
      resampling_time_range=(5.0, 10.0),
      site_name="right_grasp_center",
      ranges=mdp.PoseRanges(
        pos_x=(0.0, 0.0),  # Set per-robot.
        pos_y=(0.0, 0.0),  # Set per-robot.
        pos_z=(0.0, 0.0),  # Set per-robot.
      ),
    ),
  }

  ## --------------------------------------------------------
  # Events
  ## --------------------------------------------------------

  events = {
    "reset_robot_root": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset_broadcast,
      mode="reset",
      params={
        "position_range": (0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_frictionloss": EventTermCfg(
      mode="reset",
      func=dr.dof_frictionloss,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.5, 2.0),
        "shared_random": False,
      },
    ),
  }

  ## --------------------------------------------------------
  # Rewards
  ## --------------------------------------------------------

  rewards = {
    "pos_left": RewardTermCfg(
      func=mdp.position_command_error,
      weight=-2.0,
      params={
        "site_name": "left_grasp_center",
        "command_name": "pose_command_left",
      },
    ),
    "pos_left_fine_grained": RewardTermCfg(
      func=mdp.position_command_error_tanh,
      weight=2.0,
      params={
        "site_name": "left_grasp_center",
        "command_name": "pose_command_left",
        "std": 0.05,
      },
    ),
    "ee_left_orientation": RewardTermCfg(
      func=mdp.orientation_command_error,
      weight=-0.2,
      params={
        "site_name": "left_grasp_center",
        "command_name": "pose_command_left",
      },
    ),
    "pos_right": RewardTermCfg(
      func=mdp.position_command_error,
      weight=-2.0,
      params={
        "site_name": "right_grasp_center",
        "command_name": "pose_command_right",
      },
    ),
    "pos_right_fine_grained": RewardTermCfg(
      func=mdp.position_command_error_tanh,
      weight=3.0,
      params={
        "site_name": "right_grasp_center",
        "command_name": "pose_command_right",
        "std": 0.05,
      },
    ),
    "ee_right_orientation": RewardTermCfg(
      func=mdp.orientation_command_error,
      weight=-0.2,
      params={
        "site_name": "right_grasp_center",
        "command_name": "pose_command_right",
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2_louis,
      weight=-0.003,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # Set per-robot.
      },
    ),
    "joint_vel_hinge": RewardTermCfg(
      func=mdp.joint_velocity_hinge_penalty,
      weight=-0.05,
      params={
        "max_vel": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  ## --------------------------------------------------------
  # Terminations
  ## --------------------------------------------------------

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  ## --------------------------------------------------------
  # Curriculum
  ## --------------------------------------------------------
  curriculum = {
    "action_rate_curr": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.003},
          {"step": 5_000 * 24, "weight": -0.01},
        ],
      },
    ),
    "orientation_curr_right": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "ee_right_orientation",
        "stages": [
          {"step": 0, "weight": -0.3},
          {"step": 7_500 * 24, "weight": -0.6},
        ],
      },
    ),
    "orientation_curr_left": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "ee_left_orientation",
        "stages": [
          {"step": 0, "weight": -0.3},
          {"step": 7_500 * 24, "weight": -0.6},
        ],
      },
    ),
  }

  ## --------------------------------------------------------
  # Assemble final configuration
  ## --------------------------------------------------------

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
      ),
      num_envs=1,
      # MuJoCo-Warp environments are independent worlds.  Keeping every
      # environment in the same local frame makes the fixed table, floating
      # robot, object resets, and reward targets use identical coordinates.
      env_spacing=0.0,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )


def talos_tabletop_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create phase-0 TALOS standing and tabletop reaching configuration.

  This registered environment validates the migrated model, sensors, table,
  object, and upper-body action interface.  Contact grasp, lift, and transport
  rewards are intentionally staged for subsequent task iterations.
  """
  cfg = make_tabletop_reaching_env_cfg()
  cfg.scene.entities = {
    "robot": get_talos_grasping_robot_cfg(),
    "table": get_table_cfg(),
    "object": get_object_cfg(),
  }
  cfg.scene.num_envs = 1024 if not play else 1
  cfg.scene.extent = 2.5
  cfg.sim.nconmax = 120
  cfg.viewer.body_name = "torso_2_link"

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  # The leg actuators retain their default standing targets.  The policy acts
  # only on the torso, arms, and the two real gripper motors.
  action.actuator_names = (
    r"torso_.*_joint",
    r"arm_.*_joint",
    r"gripper_(left|right)_joint",
  )
  action.scale = {
    pattern: scale
    for pattern, scale in TALOS_GRASPING_ACTION_SCALE.items()
    if pattern.startswith(("torso_", "arm_", "gripper_"))
  }
  action.preserve_order = True
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_rate_l2,
    weight=-0.003,
  )

  left_cmd = cfg.commands["pose_command_left"]
  right_cmd = cfg.commands["pose_command_right"]
  left_cmd.ranges.pos_x = (0.35, 0.80)
  left_cmd.ranges.pos_y = (0.00, 0.45)
  left_cmd.ranges.pos_z = (0.75, 1.20)
  right_cmd.ranges.pos_x = (0.35, 0.80)
  right_cmd.ranges.pos_y = (-0.45, 0.00)
  right_cmd.ranges.pos_z = (0.75, 1.20)

  object_terms = {
    "object_pos_b": ObservationTermCfg(func=mdp.object_position_in_robot_frame),
    "object_from_left_hand": ObservationTermCfg(
      func=mdp.object_vector_from_site,
      params={"site_name": "left_grasp_center"},
    ),
    "object_from_right_hand": ObservationTermCfg(
      func=mdp.object_vector_from_site,
      params={"site_name": "right_grasp_center"},
    ),
  }
  cfg.observations["actor"].terms.update(object_terms)
  cfg.observations["critic"].terms.update(object_terms)

  lower_body = SceneEntityCfg("robot", joint_names=(r"leg_.*_joint",))
  cfg.rewards.update(
    {
      "upright": RewardTermCfg(
        func=mdp.upright,
        weight=2.0,
        params={
          "std": 0.35,
          "asset_cfg": SceneEntityCfg("robot", body_names=("torso_2_link",)),
        },
      ),
      "base_motion": RewardTermCfg(func=mdp.base_motion_l2, weight=-0.20),
      "lower_body_pose": RewardTermCfg(
        func=mdp.joint_deviation_l2,
        weight=-2.0,
        params={"asset_cfg": lower_body},
      ),
      "approach_object_left": RewardTermCfg(
        func=mdp.site_object_distance_tanh,
        weight=1.0,
        params={"site_name": "left_grasp_center", "std": 0.20},
      ),
      "approach_object_right": RewardTermCfg(
        func=mdp.site_object_distance_tanh,
        weight=1.0,
        params={"site_name": "right_grasp_center", "std": 0.20},
      ),
    }
  )

  cfg.scene.sensors = (
    ContactSensorCfg(
      name="left_gripper_object_contact",
      primary=ContactMatch(
        mode="body", pattern=r"^gripper_left_.*_link$", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="object", entity="object"),
      fields=("found", "force"),
      reduce="none",
      num_slots=8,
    ),
    ContactSensorCfg(
      name="right_gripper_object_contact",
      primary=ContactMatch(
        mode="body", pattern=r"^gripper_right_.*_link$", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="object", entity="object"),
      fields=("found", "force"),
      reduce="none",
      num_slots=8,
    ),
  )
  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.episode_length_s = int(1e9)
  return cfg


def talos_tabletop_grasp_env_cfg(
  object_shape: ObjectShape = "cube",
  play: bool = False,
  control_mode: ControlMode = "position_tracking",
) -> ManagerBasedRlEnvCfg:
  """Create phase-1 right-hand grasping with exact object-center input.

  The actor and critic both receive the simulator-truth object center in the
  robot root frame.  The term is intentionally named ``privileged_*`` so a
  later camera estimator can replace the actor-side source without changing
  the three-dimensional policy interface.
  """
  cfg = make_tabletop_reaching_env_cfg()
  robot_cfg = get_talos_grasping_robot_cfg()
  robot_cfg.init_state.pos = ROBOT_SPAWN_POSITION_M
  cfg.scene.entities = {
    "robot": robot_cfg,
    "table": get_table_cfg(),
    "object": get_object_cfg(object_shape),
  }
  cfg.scene.num_envs = 2048 if not play else 1
  cfg.scene.extent = 2.5
  cfg.sim.nconmax = None
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.viewer.body_name = "torso_2_link"
  cfg.viewer.distance = 6.0
  cfg.episode_length_s = 30.0

  initial_center_height = object_initial_position(object_shape)[2]
  object_half_extents = CUBE_HALF_SIZE_M if object_shape == "cube" else None
  sphere_radius = SPHERE_RADIUS_M if object_shape == "sphere" else None
  placement_target = (
    TARGET_ZONE_CENTER_M[0],
    TARGET_ZONE_CENTER_M[1],
    initial_center_height,
  )

  # No random Cartesian reach target is needed: the object center is the goal.
  cfg.commands = {}
  for group in cfg.observations.values():
    group.terms.pop("pose_command_left", None)
    group.terms.pop("pose_command_right", None)

  actor_terms = cfg.observations["actor"].terms
  actor_terms.update(
    {
      "base_ang_vel": ObservationTermCfg(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-0.10, n_max=0.10),
      ),
      "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.03, n_max=0.03),
      ),
      "privileged_object_center_b": ObservationTermCfg(
        func=mdp.object_position_in_robot_frame,
      ),
      "table_approach_target_b": ObservationTermCfg(
        func=mdp.point_in_robot_frame,
        params={
          "point": (
            TABLE_APPROACH_BASE_XY_M[0],
            TABLE_APPROACH_BASE_XY_M[1],
            ROBOT_SPAWN_POSITION_M[2],
          )
        },
      ),
      "placement_target_b": ObservationTermCfg(
        func=mdp.point_in_robot_frame,
        params={"point": placement_target},
      ),
    }
  )
  cfg.observations["critic"].terms.update(actor_terms)
  cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
    func=mdp.base_lin_vel
  )
  # Keep the original actor/critic columns unchanged and append the two new
  # dimensions.  This makes balance-checkpoint migration an exact zero-column
  # expansion of the first policy layer.
  phase_term = ObservationTermCfg(
    func=mdp.locomotion_phase,
    params={"period": LOCOMOTION_PERIOD_S, "control_mode": control_mode},
  )
  cfg.observations["actor"].terms["locomotion_phase"] = phase_term
  cfg.observations["critic"].terms["locomotion_phase"] = ObservationTermCfg(
    func=mdp.locomotion_phase,
    params={"period": LOCOMOTION_PERIOD_S, "control_mode": control_mode},
  )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  # Legs are policy-controlled: the robot must balance rather than relying on
  # a welded base or fixed lower-body targets.  Head and left gripper stay at
  # their defaults; both arms remain available for reach and counterbalance.
  action.actuator_names = (
    r"leg_.*_joint",
    r"torso_.*_joint",
    r"arm_.*_joint",
    r"gripper_right_joint",
  )
  action.scale = {
    pattern: scale
    for pattern, scale in TALOS_GRASPING_ACTION_SCALE.items()
    if pattern.startswith(("leg_", "torso_", "arm_", "gripper_right"))
  }
  action.preserve_order = True

  cfg.events["reset_robot_joints"].params.update(
    {
      # Match the validated WBC grasp setup exactly during the initial
      # balance curriculum instead of perturbing the unsupported landing.
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
    }
  )
  cfg.events["reset_table_root"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_root_state_uniform,
    params={
      "asset_cfg": SceneEntityCfg("table", body_names=("table",)),
      "pose_range": {},
      "velocity_range": {},
    },
  )
  cfg.events["reset_object_on_pickup_zone"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_root_state_uniform,
    params={
      "asset_cfg": SceneEntityCfg("object", body_names=("object",)),
      "pose_range": {
        "x": (-0.06, 0.06),
        "y": (-0.06, 0.06),
        "yaw": (-math.pi, math.pi),
      },
      "velocity_range": {},
    },
  )

  right_hand_pattern = (
    "^("
    + "|".join(name for name in TALOS_GRIPPER_CONTACT_BODY_NAMES if "_right_" in name)
    + ")$"
  )
  right_contact = ContactSensorCfg(
    name="right_gripper_object_contact",
    primary=ContactMatch(mode="body", pattern=right_hand_pattern, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="object", entity="object"),
    fields=("found", "force", "pos", "normal", "tangent"),
    reduce="none",
    num_slots=4,
    global_frame=True,
    track_air_time=True,
    history_length=cfg.decimation,
  )
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(leg_left_6_link|leg_right_6_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Feet and ankle bodies must remain legal terrain contacts.  The previous
  # broad ``leg_.*_link`` expression included both feet and reset every world
  # as soon as the robot landed after initialization.
  illegal_ground_pattern = (
    r"^(base_link|torso_.*_link|head_.*_link|leg_.*_4_link|arm_.*_(5|7)_link)$"
  )
  illegal_table_pattern = (
    r"^(base_link|torso_.*_link|head_.*_link|leg_.*_link|arm_.*_link)$"
  )
  body_ground = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=illegal_ground_pattern,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  body_table = ContactSensorCfg(
    name="body_table_contact",
    primary=ContactMatch(
      mode="body",
      pattern=illegal_table_pattern,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="table", entity="table"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (right_contact, feet_ground, body_ground, body_table)

  lower_body = SceneEntityCfg("robot", joint_names=(r"leg_.*_joint",))
  lower_body_lateral = SceneEntityCfg("robot", joint_names=(r"leg_.*_(1|2)_joint",))
  foot_sites = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
  left_arm = SceneEntityCfg("robot", joint_names=(r"arm_left_.*_joint",))
  right_arm = SceneEntityCfg("robot", joint_names=(r"arm_right_.*_joint",))
  right_gripper_joint = SceneEntityCfg("robot", joint_names=(r"gripper_right_joint",))
  cfg.rewards = {
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=4.0,
      params={
        "std": 0.30,
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_2_link",)),
      },
    ),
    "both_feet_contact": RewardTermCfg(
      func=mdp.both_feet_contact,
      weight=2.0,
      params={"sensor_name": feet_ground.name},
    ),
    "both_feet_near_target": RewardTermCfg(
      func=mdp.both_feet_contact_near_target,
      weight=0.0,
      params={
        "sensor_name": feet_ground.name,
        "target_position": TABLE_APPROACH_BASE_XY_M,
        "distance_threshold": 0.35,
      },
    ),
    "standing_success": RewardTermCfg(
      func=mdp.sustained_standing_success,
      weight=5.0,
      params={
        "sensor_name": feet_ground.name,
        "required_duration_s": 3.0,
        "maximum_projected_gravity_xy": 0.20,
        "maximum_linear_speed": 0.25,
        "maximum_angular_speed": 0.35,
      },
    ),
    "base_motion": RewardTermCfg(func=mdp.base_motion_l2, weight=-0.10),
    "base_drift": RewardTermCfg(
      func=mdp.base_position_deviation_l2,
      weight=-4.0,
    ),
    "navigate_to_table": RewardTermCfg(
      func=mdp.base_target_distance_tanh,
      weight=0.0,
      params={"target_position": TABLE_APPROACH_BASE_XY_M, "std": 3.0},
    ),
    "navigation_progress": RewardTermCfg(
      func=mdp.base_target_progress,
      weight=0.0,
      params={
        "target_position": TABLE_APPROACH_BASE_XY_M,
        "heading_target_position": (TABLE_CENTER_X_M, 0.0),
        "maximum_speed": 0.25,
        "maximum_projected_gravity_xy": 0.25,
        "sensor_name": feet_ground.name,
        "minimum_height": 0.85,
      },
    ),
    "navigation_speed": RewardTermCfg(
      func=mdp.base_target_speed_above_threshold,
      weight=0.0,
      params={
        "target_position": TABLE_APPROACH_BASE_XY_M,
        "heading_target_position": (TABLE_CENTER_X_M, 0.0),
        "minimum_speed": 0.05,
        "target_speed": 0.20,
        "maximum_projected_gravity_xy": 0.25,
        "sensor_name": feet_ground.name,
        "minimum_height": 0.85,
      },
    ),
    "face_table": RewardTermCfg(
      func=mdp.base_facing_target_exp,
      weight=0.0,
      params={
        "target_position": (TABLE_CENTER_X_M, 0.0),
        "std": math.radians(20.0),
      },
    ),
    "lateral_base_velocity": RewardTermCfg(
      func=mdp.base_lateral_velocity_l2,
      weight=0.0,
    ),
    "gait_contact": RewardTermCfg(
      func=mdp.feet_gait_contact_tracking,
      weight=0.0,
      params={
        "sensor_name": feet_ground.name,
        "target_position": TABLE_APPROACH_BASE_XY_M,
        "period": LOCOMOTION_PERIOD_S,
        "offsets": (0.0, 0.5),
        "stance_ratio": 0.60,
        "stop_distance": 0.35,
      },
    ),
    "swing_peak_height": RewardTermCfg(
      func=mdp.feet_swing_peak_height,
      weight=0.0,
      params={
        "sensor_name": feet_ground.name,
        "target_position": TABLE_APPROACH_BASE_XY_M,
        "target_height": FOOT_SWING_TARGET_HEIGHT_M,
        "std": 0.025,
        "minimum_air_time": 0.08,
        "stop_distance": 0.35,
        "asset_cfg": foot_sites,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.contact_foot_slip_l2,
      weight=0.0,
      params={"sensor_name": feet_ground.name, "asset_cfg": foot_sites},
    ),
    "healthy_gait_success": RewardTermCfg(
      func=mdp.sustained_healthy_gait_success,
      weight=0.0,
      params={
        "sensor_name": feet_ground.name,
        "target_position": TABLE_APPROACH_BASE_XY_M,
        "heading_target_position": (TABLE_CENTER_X_M, 0.0),
        "required_duration_s": 0.5,
        "minimum_radial_speed": 0.08,
        "maximum_radial_speed": 0.30,
        "maximum_projected_gravity_xy": 0.25,
        "maximum_heading_error": math.radians(20.0),
        "maximum_contact_slip_speed": 0.08,
        "minimum_height": 0.85,
        "minimum_step_air_time": 0.08,
        "minimum_swing_peak_height": 0.075,
        "maximum_swing_peak_height": 0.105,
        "gait_period": LOCOMOTION_PERIOD_S,
        "gait_offsets": (0.0, 0.5),
        "gait_stance_ratio": 0.60,
        "gait_match_window_s": 1.60,
        "minimum_gait_contact_match": 0.75,
        "foot_asset_cfg": foot_sites,
      },
    ),
    "reach_table_success": RewardTermCfg(
      func=mdp.sustained_navigation_success,
      weight=0.0,
      params={
        "sensor_name": feet_ground.name,
        "target_position": TABLE_APPROACH_BASE_XY_M,
        "distance_threshold": 0.35,
        "required_duration_s": 1.0,
        "maximum_projected_gravity_xy": 0.25,
        "maximum_linear_speed": 0.30,
        "maximum_angular_speed": 0.40,
        "heading_target_position": (TABLE_CENTER_X_M, 0.0),
        "maximum_heading_error": math.radians(15.0),
        "maximum_contact_slip_speed": 0.05,
        "minimum_step_air_time": 0.08,
        "foot_asset_cfg": foot_sites,
      },
    ),
    "lower_body_pose": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.5,
      params={"asset_cfg": lower_body},
    ),
    "lower_body_lateral_pose": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=0.0,
      params={"asset_cfg": lower_body_lateral},
    ),
    "left_arm_pose": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.5,
      params={"asset_cfg": left_arm},
    ),
    "right_arm_pose": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.5,
      params={"asset_cfg": right_arm},
    ),
    "approach_object": RewardTermCfg(
      func=mdp.site_object_distance_tanh,
      weight=0.0,
      params={"site_name": "right_grasp_center", "std": 0.15},
    ),
    "approach_progress": RewardTermCfg(
      func=mdp.site_object_record_progress,
      weight=0.0,
      params={"site_name": "right_grasp_center", "std": 0.15},
    ),
    "multi_link_contact": RewardTermCfg(
      func=mdp.gripper_object_contact,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "contacts_for_full_reward": 2,
      },
    ),
    "face_contact_area": RewardTermCfg(
      func=mdp.gripper_face_contact_area,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "object_half_extents": object_half_extents,
      },
    ),
    "force_closure": RewardTermCfg(
      func=mdp.gripper_force_closure,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "object_half_extents": object_half_extents,
      },
    ),
    "grasp_quality_progress": RewardTermCfg(
      func=mdp.grasp_quality_record_progress,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "object_half_extents": object_half_extents,
      },
    ),
    "face_grasp_success": RewardTermCfg(
      func=mdp.stable_face_grasp_success,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "object_half_extents": object_half_extents,
      },
    ),
    "grasp_ready_success": RewardTermCfg(
      func=mdp.sustained_grasp_ready_success,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "object_half_extents": object_half_extents,
        "required_duration_s": 0.25,
        "minimum_grasp_quality": 0.20,
      },
    ),
    "lift_progress": RewardTermCfg(
      func=mdp.object_lift_record_progress,
      weight=0.0,
      params={
        "initial_center_height": initial_center_height,
        "table_height": TABLE_TOP_HEIGHT_M,
        "target_lift_height": 0.10,
        "sensor_name": right_contact.name,
        "object_half_extents": object_half_extents,
        "sphere_radius": sphere_radius,
        "minimum_grasp_quality": 0.10,
      },
    ),
    "low_lift_success": RewardTermCfg(
      func=mdp.sustained_verified_pick_success,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "site_name": "right_grasp_center",
        "initial_center_height": initial_center_height,
        "table_height": TABLE_TOP_HEIGHT_M,
        "minimum_lift_height": 0.02,
        "required_duration_s": 0.50,
        "minimum_grasp_quality": 0.15,
        "maximum_relative_speed": 0.08,
        "maximum_object_speed": 0.12,
        "metric_prefix": "micro_pick",
        "object_half_extents": object_half_extents,
        "sphere_radius": sphere_radius,
      },
    ),
    "grasp_lift_success": RewardTermCfg(
      func=mdp.sustained_verified_pick_success,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "site_name": "right_grasp_center",
        "initial_center_height": initial_center_height,
        "table_height": TABLE_TOP_HEIGHT_M,
        "minimum_lift_height": 0.08,
        "required_duration_s": 1.0,
        "minimum_grasp_quality": 0.15,
        "maximum_relative_speed": 0.05,
        "maximum_object_speed": 0.10,
        "metric_prefix": "verified_pick",
        "object_half_extents": object_half_extents,
        "sphere_radius": sphere_radius,
      },
    ),
    "gripper_reopening": RewardTermCfg(
      func=mdp.gripper_reopening_during_contact,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "asset_cfg": right_gripper_joint,
      },
    ),
    "object_launch": RewardTermCfg(
      func=mdp.object_launch_velocity,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "initial_center_height": initial_center_height,
        "maximum_upward_speed": 0.25,
        "object_half_extents": object_half_extents,
      },
    ),
    "task_time": RewardTermCfg(func=mdp.task_time, weight=0.0),
    "excessive_object_height": RewardTermCfg(
      func=mdp.excessive_object_lift_height,
      weight=0.0,
      params={
        "initial_center_height": initial_center_height,
        "maximum_center_lift_height": 0.30,
        "excess_scale": 0.10,
      },
    ),
    "object_target": RewardTermCfg(
      func=mdp.object_target_distance_tanh,
      weight=0.0,
      params={"target_position": placement_target, "std": 0.20},
    ),
    "place_success": RewardTermCfg(
      func=mdp.place_object_success,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "initial_center_height": initial_center_height,
        "minimum_lift_height": 0.08,
        "target_position": placement_target,
        "target_xy_tolerance": 0.08,
        "target_height_tolerance": 0.04,
        "maximum_release_contacts": 0,
      },
    ),
    "termination_penalty": RewardTermCfg(
      func=mdp.is_terminated,
      weight=-200.0,
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_magnitude_l2": RewardTermCfg(
      func=mdp.action_magnitude_l2,
      weight=-0.02,
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    "joint_vel_hinge": RewardTermCfg(
      func=mdp.joint_velocity_hinge_penalty,
      weight=-0.02,
      params={
        "max_vel": 1.0,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fallen_height": TerminationTermCfg(
      func=mdp.root_height_below,
      params={"minimum_height": 0.72},
    ),
    "fallen_tilt": TerminationTermCfg(
      func=mdp.root_tilt_exceeded,
      params={"maximum_projected_gravity_xy": 0.70},
    ),
    "body_ground_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": body_ground.name},
    ),
    "body_table_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": body_table.name},
    ),
    "object_lost": TerminationTermCfg(
      func=mdp.object_height_below,
      params={"minimum_height": -10.0},
    ),
  }
  # The stages are gated only by completed-episode success rates.  Every stage
  # explicitly sets all mutable weights so a promoted environment cannot retain
  # stale shaping from the previous stage.
  safety_weights = {
    "termination_penalty": -200.0,
    "dof_pos_limits": -1.0,
    "action_magnitude_l2": -0.02,
    "action_rate_l2": -0.02,
    "joint_vel_hinge": -0.02,
  }
  stage_reward_weights = (
    {
      **safety_weights,
      "upright": 4.0,
      "both_feet_contact": 2.0,
      "both_feet_near_target": 0.0,
      "standing_success": 5.0,
      "base_motion": -0.10,
      "base_drift": -4.0,
      "navigate_to_table": 0.0,
      "navigation_progress": 0.0,
      "navigation_speed": 0.0,
      "face_table": 0.0,
      "lateral_base_velocity": 0.0,
      "gait_contact": 0.0,
      "swing_peak_height": 0.0,
      "foot_slip": -0.2,
      "healthy_gait_success": 0.0,
      "reach_table_success": 0.0,
      "lower_body_pose": -0.5,
      "lower_body_lateral_pose": -0.5,
      "left_arm_pose": -0.5,
      "right_arm_pose": -0.5,
      "approach_object": 0.0,
      "multi_link_contact": 0.0,
      "face_contact_area": 0.0,
      "force_closure": 0.0,
      "face_grasp_success": 0.0,
      "lift_progress": 0.0,
      "low_lift_success": 0.0,
      "grasp_lift_success": 0.0,
      "excessive_object_height": 0.0,
      "object_target": 0.0,
      "place_success": 0.0,
    },
    {
      **safety_weights,
      "dof_pos_limits": -1.0,
      "action_magnitude_l2": -0.005,
      "action_rate_l2": -0.01,
      "joint_vel_hinge": -0.01,
      "termination_penalty": -1000.0,
      "upright": 1.5,
      "both_feet_contact": 0.0,
      "both_feet_near_target": 0.5,
      "standing_success": 0.1,
      "base_motion": 0.0,
      "base_drift": 0.0,
      "navigate_to_table": 0.0,
      "navigation_progress": 8.0,
      "navigation_speed": 1.0,
      "face_table": 2.0,
      "lateral_base_velocity": -1.0,
      "gait_contact": 0.5,
      "swing_peak_height": 5.0,
      "foot_slip": -0.5,
      "healthy_gait_success": 1.0,
      "reach_table_success": 10.0,
      "lower_body_pose": -0.05,
      "lower_body_lateral_pose": -0.2,
      "left_arm_pose": -0.1,
      "right_arm_pose": -0.1,
      "approach_object": 0.0,
      "multi_link_contact": 0.0,
      "face_contact_area": 0.0,
      "force_closure": 0.0,
      "face_grasp_success": 0.0,
      "lift_progress": 0.0,
      "low_lift_success": 0.0,
      "grasp_lift_success": 0.0,
      "excessive_object_height": 0.0,
      "object_target": 0.0,
      "place_success": 0.0,
    },
    {
      **safety_weights,
      "upright": 2.0,
      "both_feet_contact": 1.0,
      "both_feet_near_target": 0.0,
      "standing_success": 1.0,
      "base_motion": -0.05,
      "base_drift": 0.0,
      "navigate_to_table": 0.0,
      "navigation_progress": 1.0,
      "navigation_speed": 0.5,
      "face_table": 0.5,
      "lateral_base_velocity": -0.5,
      "gait_contact": 0.0,
      "swing_peak_height": 0.0,
      "foot_slip": -0.2,
      "healthy_gait_success": 0.0,
      "reach_table_success": 1.0,
      "lower_body_pose": -0.2,
      "lower_body_lateral_pose": -0.5,
      "left_arm_pose": -0.2,
      "right_arm_pose": -0.1,
      "approach_object": 4.0,
      "multi_link_contact": 3.0,
      "face_contact_area": 0.0,
      "force_closure": 0.0,
      "face_grasp_success": 0.0,
      "lift_progress": 8.0,
      "low_lift_success": 0.0,
      "grasp_lift_success": 6.0,
      "excessive_object_height": -10.0,
      "object_target": 0.0,
      "place_success": 0.0,
    },
    {
      **safety_weights,
      "upright": 2.0,
      "both_feet_contact": 1.0,
      "both_feet_near_target": 0.0,
      "standing_success": 1.0,
      "base_motion": -0.05,
      "base_drift": 0.0,
      "navigate_to_table": 0.0,
      "navigation_progress": 0.5,
      "navigation_speed": 0.25,
      "face_table": 0.5,
      "lateral_base_velocity": -0.5,
      "gait_contact": 0.0,
      "swing_peak_height": 0.0,
      "foot_slip": -0.2,
      "healthy_gait_success": 0.0,
      "reach_table_success": 0.5,
      "lower_body_pose": -0.2,
      "lower_body_lateral_pose": -0.5,
      "left_arm_pose": -0.2,
      "right_arm_pose": -0.1,
      "approach_object": 1.0,
      "multi_link_contact": 2.0,
      "face_contact_area": 0.0,
      "force_closure": 0.0,
      "face_grasp_success": 0.0,
      "lift_progress": 2.0,
      "low_lift_success": 0.0,
      "grasp_lift_success": 2.0,
      "excessive_object_height": -10.0,
      "object_target": 5.0,
      "place_success": 12.0,
    },
  )
  cfg.curriculum = {
    "task_stage": CurriculumTermCfg(
      func=mdp.performance_stage_curriculum,
      params={
        "stage_reward_weights": stage_reward_weights,
        "promotion_reward_names": (
          "standing_success",
          "reach_table_success",
          "grasp_lift_success",
        ),
        "promotion_success_rates": (0.80, 0.70, 0.60),
        "evaluation_episodes": (4096, 4096, 4096),
        "initial_stage": 0,
        "stage_termination_params": (
          {"object_lost": {"minimum_height": -10.0}},
          {"object_lost": {"minimum_height": -10.0}},
          {
            "object_lost": {
              "minimum_height": TABLE_TOP_HEIGHT_M - 0.12,
            }
          },
          {
            "object_lost": {
              "minimum_height": TABLE_TOP_HEIGHT_M - 0.12,
            }
          },
        ),
      },
    )
  }

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
  return cfg


def talos_tabletop_position_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Train only the position-tracking mode before manipulation is introduced.

  Stage 0 must demonstrate a healthy alternating gait.  Stage 1 learns the
  complete three-metre approach and remains the terminal curriculum stage, so
  this task cannot silently advance into grasp/manipulation rewards.
  """
  cfg = talos_tabletop_grasp_env_cfg(
    object_shape="cube",
    play=play,
    control_mode="position_tracking",
  )
  full_params = cfg.curriculum["task_stage"].params
  navigation_weights = dict(full_params["stage_reward_weights"][1])
  gait_acquisition_weights = {
    **navigation_weights,
    "upright": 2.0,
    "both_feet_near_target": 0.0,
    "standing_success": 0.0,
    "navigation_progress": 4.0,
    "navigation_speed": 1.0,
    "face_table": 2.0,
    "lateral_base_velocity": -1.0,
    "gait_contact": 0.75,
    "swing_peak_height": 5.0,
    "foot_slip": -0.75,
    "healthy_gait_success": 10.0,
    "reach_table_success": 0.0,
    "lower_body_lateral_pose": -0.3,
  }
  position_tracking_weights = {
    **navigation_weights,
    "both_feet_contact": 0.0,
    "both_feet_near_target": 0.5,
    "healthy_gait_success": 1.0,
  }
  stage_reward_weights = (
    gait_acquisition_weights,
    position_tracking_weights,
  )
  for reward_name, weight in gait_acquisition_weights.items():
    cfg.rewards[reward_name].weight = weight
  cfg.curriculum = {
    "task_stage": CurriculumTermCfg(
      func=mdp.performance_stage_curriculum,
      params={
        "stage_reward_weights": stage_reward_weights,
        "promotion_reward_names": ("healthy_gait_success",),
        "promotion_success_rates": (0.70,),
        "evaluation_episodes": (4096,),
        "initial_stage": 0,
        "stage_termination_params": (
          {"object_lost": {"minimum_height": -10.0}},
          {"object_lost": {"minimum_height": -10.0}},
        ),
      },
    )
  }
  return cfg


def talos_tabletop_stationary_grasp_env_cfg(
  object_shape: ObjectShape = "cube",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Train a stationary manipulation policy without navigation or placement.

  Stage 0 removes the table and object while the robot learns to stand still.
  Stage 1 restores the scene and learns a paired-face, wrench-stable grasp.
  Stage 2 requires a verified 2 cm micro-pick and hold test.  Stage 3 performs
  the full 8--10 cm pick.  Transport and placement remain disabled.
  """
  cfg = talos_tabletop_grasp_env_cfg(
    object_shape=object_shape,
    play=play,
    control_mode="manipulation",
  )
  cfg.scene.entities["robot"].init_state.pos = STATIONARY_GRASP_ROBOT_POSITION_M

  for group in cfg.observations.values():
    obstacle_params = dict(group.terms["collision_obstacle_boxes_b"].params)
    obstacle_params["minimum_curriculum_stage"] = 1
    group.terms["collision_obstacle_boxes_b"] = ObservationTermCfg(
      func=mdp.obstacle_boxes_in_robot_frame,
      params=obstacle_params,
    )
    group.terms["privileged_object_center_b"] = ObservationTermCfg(
      func=mdp.object_position_in_robot_frame,
      params={"minimum_curriculum_stage": 1},
    )
    group.terms["table_approach_target_b"] = ObservationTermCfg(
      func=mdp.zero_vector,
      params={"size": 3},
    )
    group.terms["placement_target_b"] = ObservationTermCfg(
      func=mdp.zero_vector,
      params={"size": 3},
    )

  cfg.rewards["standing_success"].params.update(
    {
      "required_duration_s": 5.0,
      "maximum_projected_gravity_xy": 0.15,
      "maximum_linear_speed": 0.05,
      "maximum_angular_speed": 0.10,
    }
  )

  disabled_weights = {name: 0.0 for name in cfg.rewards}
  standing_weights = {
    **disabled_weights,
    "termination_penalty": -200.0,
    "dof_pos_limits": -1.0,
    "action_magnitude_l2": -0.02,
    "action_rate_l2": -0.05,
    "joint_vel_hinge": -0.02,
    "upright": 4.0,
    "both_feet_contact": 2.0,
    "standing_success": 10.0,
    "base_motion": -1.0,
    "base_drift": -4.0,
    "foot_slip": -0.5,
    "lower_body_pose": -0.5,
    "lower_body_lateral_pose": -0.5,
    "left_arm_pose": -0.5,
    "right_arm_pose": -0.5,
  }
  manipulation_balance_weights = {
    **disabled_weights,
    "termination_penalty": -200.0,
    "dof_pos_limits": -1.0,
    "action_magnitude_l2": -0.02,
    "action_rate_l2": -0.05,
    "joint_vel_hinge": -0.02,
    "upright": 3.0,
    "both_feet_contact": 1.0,
    "standing_success": 1.0,
    "base_motion": -0.5,
    "base_drift": -4.0,
    "foot_slip": -0.5,
    "lower_body_pose": -0.3,
    "lower_body_lateral_pose": -0.5,
    "left_arm_pose": -0.5,
    "right_arm_pose": -0.05,
    "excessive_object_height": -10.0,
  }
  grasp_acquisition_weights = {
    **manipulation_balance_weights,
    "approach_progress": 4.0,
    "grasp_quality_progress": 12.0,
    "grasp_ready_success": 25.0,
    "gripper_reopening": -1.0,
    "object_launch": -5.0,
    "task_time": -0.02,
  }
  micro_pick_weights = {
    **manipulation_balance_weights,
    "approach_progress": 1.0,
    "grasp_quality_progress": 8.0,
    "grasp_ready_success": 2.0,
    "lift_progress": 30.0,
    "low_lift_success": 50.0,
    "gripper_reopening": -1.0,
    "object_launch": -10.0,
    "task_time": -0.02,
  }
  full_pick_weights = {
    **manipulation_balance_weights,
    "approach_progress": 0.5,
    "grasp_quality_progress": 5.0,
    "grasp_ready_success": 1.0,
    "lift_progress": 50.0,
    "low_lift_success": 2.0,
    "grasp_lift_success": 80.0,
    "gripper_reopening": -1.0,
    "object_launch": -15.0,
    "task_time": -0.02,
    "excessive_object_height": -20.0,
  }
  for reward_name, weight in standing_weights.items():
    cfg.rewards[reward_name].weight = weight

  # Keep the inactive scene upright and physically well-conditioned on its own
  # support surface.  Moving it underground would penetrate MuJoCo's infinite
  # ground plane; moving it laterally removes it from the robot without that
  # contact artifact.
  hidden_pose_range = {"x": (HIDDEN_SCENE_X_OFFSET_M, HIDDEN_SCENE_X_OFFSET_M)}
  active_object_pose_range = {
    "x": (-0.06, 0.06),
    "y": (-0.06, 0.06),
    "yaw": (-math.pi, math.pi),
  }
  cfg.curriculum = {
    "task_stage": CurriculumTermCfg(
      func=mdp.performance_stage_curriculum,
      params={
        "stage_reward_weights": (
          standing_weights,
          grasp_acquisition_weights,
          micro_pick_weights,
          full_pick_weights,
        ),
        "promotion_reward_names": (
          "standing_success",
          "grasp_ready_success",
          "low_lift_success",
        ),
        "promotion_success_rates": (0.80, 0.45, 0.40),
        "evaluation_episodes": (4096, 4096, 4096),
        "initial_stage": 0,
        "stage_termination_params": (
          {"object_lost": {"minimum_height": -100.0}},
          {
            "object_lost": {
              "minimum_height": TABLE_TOP_HEIGHT_M - 0.12,
            }
          },
          {
            "object_lost": {
              "minimum_height": TABLE_TOP_HEIGHT_M - 0.12,
            }
          },
          {
            "object_lost": {
              "minimum_height": TABLE_TOP_HEIGHT_M - 0.12,
            }
          },
        ),
        "stage_event_params": (
          {
            "reset_table_root": {"pose_range": hidden_pose_range},
            "reset_object_on_pickup_zone": {"pose_range": hidden_pose_range},
          },
          {
            "reset_table_root": {"pose_range": {}},
            "reset_object_on_pickup_zone": {"pose_range": active_object_pose_range},
          },
          {
            "reset_table_root": {"pose_range": {}},
            "reset_object_on_pickup_zone": {"pose_range": active_object_pose_range},
          },
          {
            "reset_table_root": {"pose_range": {}},
            "reset_object_on_pickup_zone": {"pose_range": active_object_pose_range},
          },
        ),
      },
    )
  }
  return cfg


def talos_tabletop_stable_contact_lift_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Train an 8 cm airborne force-closure hold from an IK contact seed."""
  cfg = talos_tabletop_stationary_grasp_env_cfg(
    object_shape="cube",
    play=play,
  )

  robot_cfg = cfg.scene.entities["robot"]
  joint_pos = dict(robot_cfg.init_state.joint_pos or {})
  # Replace shared regex entries so the IK values cannot be shadowed during
  # initial-state name resolution.
  for pattern in tuple(joint_pos):
    if pattern.startswith("arm_.*_") or pattern.startswith("gripper_.*_"):
      joint_pos.pop(pattern)
    elif pattern.startswith("gripper_(left|right)"):
      joint_pos.pop(pattern)
  joint_pos.update(
    {
      "arm_left_4_joint": -1.5,
      "arm_left_5_joint": 0.0,
      "arm_left_6_joint": 0.0,
      "arm_left_7_joint": 0.0,
      **STABLE_HOLD_RIGHT_ARM_JOINT_POS,
    }
  )
  for side, main_position in (
    ("left", -0.24),
    ("right", STABLE_HOLD_GRIPPER_POSITION),
  ):
    joint_pos.update(
      {
        f"gripper_{side}_joint": main_position,
        f"gripper_{side}_inner_double_joint": main_position,
        f"gripper_{side}_motor_single_joint": -main_position,
        f"gripper_{side}_inner_single_joint": -main_position,
        f"gripper_{side}_fingertip_1_joint": -main_position,
        f"gripper_{side}_fingertip_2_joint": -main_position,
        f"gripper_{side}_fingertip_3_joint": -main_position,
      }
    )
  robot_cfg.init_state.joint_pos = joint_pos

  # A true reverse curriculum begins from the completed skill.  The cube is a
  # six-DoF free body at the 8 cm target; no weld or kinematic attachment is
  # introduced.  The nonzero yaw and lateral offset place the cube inside the
  # useful three-finger closing envelope instead of the old fully-open seed.
  cfg.scene.entities["object"].init_state.pos = STABLE_HOLD_OBJECT_POSITION_M
  cfg.scene.entities["object"].init_state.rot = STABLE_HOLD_OBJECT_ROT_WXYZ
  cfg.events["reset_robot_joints"].params.update(
    {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
  )
  cfg.events["reset_object_on_pickup_zone"].params.update(
    {"pose_range": {}, "velocity_range": {}}
  )
  cfg.curriculum = {}

  # Stationary-task observations were hidden in curriculum stage zero.  Make
  # them unconditionally visible while preserving the exact 222/225 network
  # dimensions of the reusable standing checkpoint.
  for group in cfg.observations.values():
    obstacle_params = dict(group.terms["collision_obstacle_boxes_b"].params)
    obstacle_params.pop("minimum_curriculum_stage", None)
    group.terms["collision_obstacle_boxes_b"] = ObservationTermCfg(
      func=mdp.obstacle_boxes_in_robot_frame,
      params=obstacle_params,
    )
    group.terms["privileged_object_center_b"] = ObservationTermCfg(
      func=mdp.object_position_in_robot_frame,
    )

  disabled_weights = {name: 0.0 for name in cfg.rewards}
  lift_weights = {
    **disabled_weights,
    "termination_penalty": -200.0,
    "dof_pos_limits": -1.0,
    "action_magnitude_l2": -0.01,
    "action_rate_l2": -0.03,
    "joint_vel_hinge": -0.02,
    "upright": 3.0,
    "both_feet_contact": 1.0,
    "standing_success": 1.0,
    "base_motion": -0.5,
    "base_drift": -4.0,
    "foot_slip": -0.5,
    "lower_body_pose": -0.3,
    "lower_body_lateral_pose": -0.5,
    "left_arm_pose": -0.5,
    "right_arm_pose": -0.02,
    "multi_link_contact": 0.25,
    "face_contact_area": 2.0,
    "force_closure": 4.0,
    "gripper_reopening": -1.0,
    "object_launch": -20.0,
    "task_time": -0.01,
    "excessive_object_height": -20.0,
  }
  for name, weight in lift_weights.items():
    cfg.rewards[name].weight = weight

  initial_center_height = object_initial_position("cube")[2]
  cfg.rewards["contact_lift_progress"] = RewardTermCfg(
    func=mdp.contact_verified_lift_height,
    weight=6.0,
    params={
      "initial_center_height": initial_center_height,
      "table_height": TABLE_TOP_HEIGHT_M,
      "target_lift_height": STABLE_HOLD_TARGET_LIFT_HEIGHT_M,
      "sensor_name": "right_gripper_object_contact",
      "minimum_contact_links": 2,
      "object_half_extents": CUBE_HALF_SIZE_M,
      "clearance_margin": 0.0,
    },
  )
  cfg.rewards["contact_height_target"] = RewardTermCfg(
    func=mdp.contact_verified_height_target,
    weight=20.0,
    params={
      "initial_center_height": initial_center_height,
      "table_height": TABLE_TOP_HEIGHT_M,
      "target_lift_height": STABLE_HOLD_TARGET_LIFT_HEIGHT_M,
      "height_error_std": 0.025,
      "object_speed_std": 0.15,
      "sensor_name": "right_gripper_object_contact",
      "minimum_contact_links": 2,
      "object_half_extents": CUBE_HALF_SIZE_M,
      "clearance_margin": 0.0,
    },
  )
  cfg.rewards["contact_lift_success"] = RewardTermCfg(
    func=mdp.sustained_verified_pick_success,
    weight=80.0,
    params={
      "sensor_name": "right_gripper_object_contact",
      "site_name": "right_grasp_center",
      "initial_center_height": initial_center_height,
      "table_height": TABLE_TOP_HEIGHT_M,
      "minimum_lift_height": 0.06,
      "target_lift_height": STABLE_HOLD_TARGET_LIFT_HEIGHT_M,
      "maximum_height_error": 0.015,
      "required_duration_s": 5.0,
      "minimum_contact_links": 2,
      "minimum_grasp_quality": 0.01,
      "maximum_relative_speed": 0.08,
      "maximum_object_speed": 0.12,
      "metric_prefix": "contact_lift",
      "object_half_extents": CUBE_HALF_SIZE_M,
      "clearance_margin": 0.0,
    },
  )
  cfg.terminations["object_lost"].params["minimum_height"] = TABLE_TOP_HEIGHT_M - 0.12
  cfg.scene.num_envs = 2048 if not play else 1
  cfg.episode_length_s = 12.0 if not play else int(1e9)
  return cfg


def talos_tabletop_anchor_standing_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Train robust standing from the complete 20-stage IK anchor distribution."""
  cfg = talos_tabletop_stable_contact_lift_env_cfg(play=play)
  anchor_bank_path = os.environ.get("TALOS_STANDING_ANCHOR_BANK", "")

  # Standing is the only objective here.  Contact/lift terms remain registered
  # to preserve the 222/225 observation and 29-action checkpoint interface, but
  # cannot influence this pretraining policy.
  standing_weights = {name: 0.0 for name in cfg.rewards}
  standing_weights.update(
    {
      "termination_penalty": -200.0,
      "dof_pos_limits": -1.0,
      "action_magnitude_l2": -0.01,
      "action_rate_l2": -0.02,
      "joint_vel_hinge": -0.02,
      "upright": 4.0,
      "both_feet_contact": 2.0,
      "standing_success": 10.0,
      "base_motion": -1.0,
      "base_drift": -4.0,
      "foot_slip": -0.5,
      "lower_body_pose": -0.5,
      "lower_body_lateral_pose": -0.5,
      "left_arm_pose": -0.1,
      "right_arm_pose": -0.02,
    }
  )
  for reward_name, weight in standing_weights.items():
    cfg.rewards[reward_name].weight = weight
  # The stationary-grasp task's 0.05/0.10 speed gate is appropriate for
  # deterministic evaluation but rejects harmless balance motion under PPO's
  # stochastic exploration.  These measured thresholds retain a strict
  # five-second, two-feet, upright test while making the training success rate
  # representative (model_200: 84.4% deterministic on Stage-0 DR).
  cfg.rewards["standing_success"].params.update(
    {
      "maximum_linear_speed": 0.10,
      "maximum_angular_speed": 0.20,
    }
  )
  cfg.terminations["object_lost"].params["minimum_height"] = -10.0
  # The generated manipulation anchors intentionally place the arm/hand close
  # to the tabletop.  Table contact is therefore diagnostic, not evidence that
  # the balance policy failed; only falling/ground contacts terminate standing.
  cfg.terminations.pop("body_table_contact", None)

  controlled_joints = SceneEntityCfg(
    "robot",
    joint_names=(
      r"leg_.*_joint",
      r"torso_.*_joint",
      r"arm_.*_joint",
      r"gripper_right_joint",
    ),
  )
  # MJLab's PD randomizer indexes actuator *groups*, whereas actuator-name
  # resolution returns individual control indices.  Randomizing all TALOS
  # groups is both physically appropriate and avoids mixing those namespaces.
  controlled_actuators = SceneEntityCfg("robot")

  # The anchor event atomically replaces the independent root/joint/object
  # resets inherited from the manipulation task.
  cfg.events.pop("reset_robot_root", None)
  cfg.events.pop("reset_robot_joints", None)
  cfg.events.pop("reset_object_on_pickup_zone", None)
  cfg.events["reset_foot_friction"] = EventTermCfg(
    mode="reset",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        geom_names=(r"left_foot_collision", r"right_foot_collision"),
      ),
      "ranges": (0.70, 1.20),
      "operation": "abs",
      "axes": [0],
      "shared_random": True,
    },
  )
  cfg.events["reset_pd_gains"] = EventTermCfg(
    mode="reset",
    func=dr.pd_gains,
    params={
      "asset_cfg": controlled_actuators,
      "kp_range": (0.95, 1.05),
      "kd_range": (0.95, 1.05),
      "operation": "scale",
    },
  )
  cfg.events["reset_encoder_bias"] = EventTermCfg(
    mode="reset",
    func=dr.encoder_bias,
    params={
      "asset_cfg": controlled_joints,
      "bias_range": (-0.002, 0.002),
    },
  )
  cfg.events["reset_anchor_state"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_from_anchor_bank,
    params={
      "anchor_bank_path": anchor_bank_path,
      "joint_position_range": (-0.01, 0.01),
      "joint_velocity_range": (-0.02, 0.02),
      "root_pose_range": {
        "x": (-0.003, 0.003),
        "y": (-0.003, 0.003),
        "roll": (-math.radians(0.5), math.radians(0.5)),
        "pitch": (-math.radians(0.5), math.radians(0.5)),
        "yaw": (-math.radians(1.0), math.radians(1.0)),
      },
      "root_velocity_range": {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (-0.02, 0.02),
        "roll": (-0.03, 0.03),
        "pitch": (-0.03, 0.03),
        "yaw": (-0.03, 0.03),
      },
      "object_pose_range": {
        "x": (-0.002, 0.002),
        "y": (-0.002, 0.002),
        "z": (-0.002, 0.002),
        "roll": (-math.radians(1.0), math.radians(1.0)),
        "pitch": (-math.radians(1.0), math.radians(1.0)),
        "yaw": (-math.radians(1.0), math.radians(1.0)),
      },
      "object_velocity_range": {
        axis: (-0.01, 0.01) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
      },
      "robot_cfg": SceneEntityCfg("robot"),
      "object_cfg": SceneEntityCfg("object"),
    },
  )
  cfg.events["push_robot"] = EventTermCfg(
    mode="interval",
    # The sparse success reward requires five uninterrupted stable seconds.
    # Schedule exactly one push after that evaluation interval so recovery is
    # still trained without making curriculum promotion logically impossible.
    interval_range_s=(6.0, 6.5),
    is_global_time=False,
    func=mdp.push_by_setting_velocity,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "velocity_range": {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.02, 0.02),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.05, 0.05),
      },
    },
  )

  anchor_ranges = (
    {
      "joint_position_range": (-0.01, 0.01),
      "joint_velocity_range": (-0.02, 0.02),
      "root_pose_range": {
        "x": (-0.003, 0.003),
        "y": (-0.003, 0.003),
        "roll": (-math.radians(0.5), math.radians(0.5)),
        "pitch": (-math.radians(0.5), math.radians(0.5)),
        "yaw": (-math.radians(1.0), math.radians(1.0)),
      },
      "root_velocity_range": {
        axis: (-value, value)
        for axis, value in {
          "x": 0.02,
          "y": 0.02,
          "z": 0.02,
          "roll": 0.03,
          "pitch": 0.03,
          "yaw": 0.03,
        }.items()
      },
      "object_pose_range": {
        "x": (-0.002, 0.002),
        "y": (-0.002, 0.002),
        "z": (-0.002, 0.002),
        "roll": (-math.radians(1.0), math.radians(1.0)),
        "pitch": (-math.radians(1.0), math.radians(1.0)),
        "yaw": (-math.radians(1.0), math.radians(1.0)),
      },
      "object_velocity_range": {
        axis: (-0.01, 0.01) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
      },
    },
    {
      "joint_position_range": (-0.03, 0.03),
      "joint_velocity_range": (-0.08, 0.08),
      "root_pose_range": {
        "x": (-0.01, 0.01),
        "y": (-0.01, 0.01),
        "roll": (-math.radians(2.0), math.radians(2.0)),
        "pitch": (-math.radians(2.0), math.radians(2.0)),
        "yaw": (-math.radians(3.0), math.radians(3.0)),
      },
      "root_velocity_range": {
        axis: (-value, value)
        for axis, value in {
          "x": 0.08,
          "y": 0.08,
          "z": 0.08,
          "roll": 0.15,
          "pitch": 0.15,
          "yaw": 0.15,
        }.items()
      },
      "object_pose_range": {
        "x": (-0.008, 0.008),
        "y": (-0.008, 0.008),
        "z": (-0.008, 0.008),
        "roll": (-math.radians(3.0), math.radians(3.0)),
        "pitch": (-math.radians(3.0), math.radians(3.0)),
        "yaw": (-math.radians(3.0), math.radians(3.0)),
      },
      "object_velocity_range": {
        axis: (-0.04, 0.04) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
      },
    },
    {
      "joint_position_range": (-0.06, 0.06),
      "joint_velocity_range": (-0.15, 0.15),
      "root_pose_range": {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "roll": (-math.radians(4.0), math.radians(4.0)),
        "pitch": (-math.radians(4.0), math.radians(4.0)),
        "yaw": (-math.radians(6.0), math.radians(6.0)),
      },
      "root_velocity_range": {
        axis: (-value, value)
        for axis, value in {
          "x": 0.15,
          "y": 0.15,
          "z": 0.15,
          "roll": 0.30,
          "pitch": 0.30,
          "yaw": 0.30,
        }.items()
      },
      "object_pose_range": {
        "x": (-0.015, 0.015),
        "y": (-0.015, 0.015),
        "z": (-0.015, 0.015),
        "roll": (-math.radians(6.0), math.radians(6.0)),
        "pitch": (-math.radians(6.0), math.radians(6.0)),
        "yaw": (-math.radians(6.0), math.radians(6.0)),
      },
      "object_velocity_range": {
        axis: (-0.08, 0.08) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
      },
    },
  )
  stage_event_params = (
    {
      "reset_anchor_state": anchor_ranges[0],
      "reset_frictionloss": {"ranges": (0.80, 1.20)},
      "reset_foot_friction": {"ranges": (0.70, 1.20)},
      "reset_pd_gains": {"kp_range": (0.95, 1.05), "kd_range": (0.95, 1.05)},
      "reset_encoder_bias": {"bias_range": (-0.002, 0.002)},
      "push_robot": {
        "velocity_range": {
          axis: (-value, value)
          for axis, value in {
            "x": 0.05,
            "y": 0.05,
            "z": 0.02,
            "roll": 0.05,
            "pitch": 0.05,
            "yaw": 0.05,
          }.items()
        }
      },
    },
    {
      "reset_anchor_state": anchor_ranges[1],
      "reset_frictionloss": {"ranges": (0.50, 1.50)},
      "reset_foot_friction": {"ranges": (0.50, 1.30)},
      "reset_pd_gains": {"kp_range": (0.85, 1.15), "kd_range": (0.85, 1.15)},
      "reset_encoder_bias": {"bias_range": (-0.005, 0.005)},
      "push_robot": {
        "velocity_range": {
          axis: (-value, value)
          for axis, value in {
            "x": 0.15,
            "y": 0.15,
            "z": 0.05,
            "roll": 0.20,
            "pitch": 0.20,
            "yaw": 0.20,
          }.items()
        }
      },
    },
    {
      "reset_anchor_state": anchor_ranges[2],
      "reset_frictionloss": {"ranges": (0.25, 2.00)},
      "reset_foot_friction": {"ranges": (0.35, 1.50)},
      "reset_pd_gains": {"kp_range": (0.70, 1.30), "kd_range": (0.70, 1.30)},
      "reset_encoder_bias": {"bias_range": (-0.010, 0.010)},
      "push_robot": {
        "velocity_range": {
          axis: (-value, value)
          for axis, value in {
            "x": 0.30,
            "y": 0.30,
            "z": 0.10,
            "roll": 0.40,
            "pitch": 0.40,
            "yaw": 0.40,
          }.items()
        }
      },
    },
  )
  cfg.curriculum = {
    "standing_randomization_stage": CurriculumTermCfg(
      func=mdp.performance_stage_curriculum,
      params={
        "stage_reward_weights": tuple(dict(standing_weights) for _ in range(3)),
        "promotion_reward_names": ("standing_success", "standing_success"),
        "promotion_success_rates": (0.80, 0.80),
        "evaluation_episodes": (8192, 8192),
        "stage_termination_params": ({}, {}, {}),
        "stage_event_params": stage_event_params,
        "initial_stage": 0,
        "strict_promotion": True,
      },
    )
  }
  cfg.scene.num_envs = 2048 if not play else 1
  cfg.episode_length_s = 10.0 if not play else int(1e9)
  return cfg
