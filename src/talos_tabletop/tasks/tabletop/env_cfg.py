"""TALOS tabletop reaching and privileged-position grasping tasks."""

import math

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
  TABLE_TOP_HEIGHT_M,
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
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
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
) -> ManagerBasedRlEnvCfg:
  """Create phase-1 right-hand grasping with exact object-center input.

  The actor and critic both receive the simulator-truth object center in the
  robot root frame.  The term is intentionally named ``privileged_*`` so a
  later camera estimator can replace the actor-side source without changing
  the three-dimensional policy interface.
  """
  cfg = make_tabletop_reaching_env_cfg()
  cfg.scene.entities = {
    "robot": get_talos_grasping_robot_cfg(),
    "table": get_table_cfg(),
    "object": get_object_cfg(object_shape),
  }
  cfg.scene.num_envs = 2048 if not play else 1
  cfg.scene.extent = 2.5
  cfg.sim.nconmax = None
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.viewer.body_name = "torso_2_link"
  cfg.episode_length_s = 10.0

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
    }
  )
  cfg.observations["critic"].terms.update(actor_terms)
  cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
    func=mdp.base_lin_vel
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
      "position_range": (-0.02, 0.02),
      "velocity_range": (0.0, 0.0),
    }
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

  right_hand_pattern = "^(" + "|".join(
    name for name in TALOS_GRIPPER_CONTACT_BODY_NAMES if "_right_" in name
  ) + ")$"
  right_contact = ContactSensorCfg(
    name="right_gripper_object_contact",
    primary=ContactMatch(mode="body", pattern=right_hand_pattern, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="object", entity="object"),
    fields=("found", "force"),
    reduce="none",
    num_slots=4,
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

  initial_center_height = object_initial_position(object_shape)[2]
  lower_body = SceneEntityCfg("robot", joint_names=(r"leg_.*_joint",))
  left_arm = SceneEntityCfg("robot", joint_names=(r"arm_left_.*_joint",))
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
    "base_motion": RewardTermCfg(func=mdp.base_motion_l2, weight=-0.10),
    "base_drift": RewardTermCfg(
      func=mdp.base_position_deviation_l2,
      weight=-4.0,
    ),
    "lower_body_pose": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.5,
      params={"asset_cfg": lower_body},
    ),
    "left_arm_pose": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.1,
      params={"asset_cfg": left_arm},
    ),
    "approach_object": RewardTermCfg(
      func=mdp.site_object_distance_tanh,
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
    "lift_progress": RewardTermCfg(
      func=mdp.object_lift_progress,
      weight=0.0,
      params={
        "initial_center_height": initial_center_height,
        "target_lift_height": 0.10,
      },
    ),
    "grasp_lift_success": RewardTermCfg(
      func=mdp.grasp_and_lift_success,
      weight=0.0,
      params={
        "sensor_name": right_contact.name,
        "initial_center_height": initial_center_height,
        "minimum_lift_height": 0.08,
        "minimum_contacts": 2,
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
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
      params={"minimum_height": TABLE_TOP_HEIGHT_M - 0.12},
    ),
  }
  # One PPO iteration collects 24 environment steps.  First learn a quiet,
  # balanced landing, then introduce reaching, contact, and finally lifting.
  steps_per_iteration = 24
  cfg.curriculum = {
    "upright": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "upright",
        "stages": [
          {"step": 0, "weight": 4.0},
          {"step": 500 * steps_per_iteration, "weight": 2.0},
        ],
      },
    ),
    "both_feet_contact": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "both_feet_contact",
        "stages": [
          {"step": 0, "weight": 2.0},
          {"step": 500 * steps_per_iteration, "weight": 0.5},
        ],
      },
    ),
    "base_drift": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "base_drift",
        "stages": [
          {"step": 0, "weight": -4.0},
          {"step": 500 * steps_per_iteration, "weight": -2.0},
        ],
      },
    ),
    "action_rate": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.05},
          {"step": 500 * steps_per_iteration, "weight": -0.02},
          {"step": 3_000 * steps_per_iteration, "weight": -0.01},
        ],
      },
    ),
    "approach_object": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "approach_object",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 500 * steps_per_iteration, "weight": 2.0},
        ],
      },
    ),
    "multi_link_contact": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "multi_link_contact",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 1_500 * steps_per_iteration, "weight": 3.0},
        ],
      },
    ),
    "lift_progress": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "lift_progress",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 3_000 * steps_per_iteration, "weight": 8.0},
        ],
      },
    ),
    "grasp_lift_success": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "grasp_lift_success",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 3_000 * steps_per_iteration, "weight": 5.0},
        ],
      },
    ),
  }

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
  return cfg
