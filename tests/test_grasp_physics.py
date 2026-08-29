import torch
from mjlab.envs import ManagerBasedRlEnv
from talos_tabletop.assets import TABLE_CENTER_X_M, TABLE_TOP_HALF_SIZE_M
from talos_tabletop.tasks.tabletop import mdp
from talos_tabletop.tasks.tabletop.env_cfg import (
  MAX_COLLISION_OBSTACLES,
  ROBOT_SPAWN_POSITION_M,
  TABLE_TOP_HEIGHT_M,
  talos_tabletop_grasp_env_cfg,
  talos_tabletop_stationary_grasp_env_cfg,
)


def test_normal_foot_landing_does_not_reset_or_move_object() -> None:
  cfg = talos_tabletop_grasp_env_cfg(object_shape="cube", play=True)
  cfg.scene.num_envs = 1
  cfg.curriculum["task_stage"].params["evaluation_episodes"] = (1, 1, 1)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=3)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    initial_object_xy = obj.data.root_link_pos_w[0, :2].clone()
    zero_action = torch.zeros((1, env.action_manager.total_action_dim))

    assert torch.allclose(
      robot.data.root_link_pos_w[0],
      torch.tensor(ROBOT_SPAWN_POSITION_M),
      atol=1.0e-6,
    )
    assert torch.count_nonzero(env.scene.env_origins) == 0

    obstacle_boxes = mdp.obstacle_boxes_in_robot_frame(
      env,
      obstacle_names=("table",),
      half_extents=(TABLE_TOP_HALF_SIZE_M,),
      max_obstacles=MAX_COLLISION_OBSTACLES,
    ).reshape(1, MAX_COLLISION_OBSTACLES, mdp.OBSTACLE_BOX_FEATURE_DIM)
    expected_table_center_b = torch.tensor(
      (
        TABLE_CENTER_X_M - ROBOT_SPAWN_POSITION_M[0],
        0.0,
        0.82 - ROBOT_SPAWN_POSITION_M[2],
      )
    )
    assert torch.allclose(obstacle_boxes[0, 0, :3], expected_table_center_b)
    assert torch.allclose(
      obstacle_boxes[0, 0, 3:6], torch.tensor(TABLE_TOP_HALF_SIZE_M)
    )
    assert obstacle_boxes[0, 0, 10] == 1.0
    assert torch.count_nonzero(obstacle_boxes[0, 1:]) == 0

    for _ in range(30):
      env.step(zero_action)
      assert not bool(env.reset_buf[0])

    assert (
      torch.linalg.vector_norm(obj.data.root_link_pos_w[0, :2] - initial_object_xy)
      < 1.0e-4
    )
    assert bool((env.scene["feet_ground_contact"].data.found[0] > 0).all())

    curriculum = env.curriculum_manager.get_term_cfg("task_stage").func
    promotion_rewards = (
      "standing_success",
      "reach_table_success",
      "grasp_lift_success",
    )
    for expected_stage, reward_name in enumerate(promotion_rewards, start=1):
      env.episode_length_buf[0] = 1
      env.reward_manager._episode_sums[reward_name][0] = 1.0
      env.curriculum_manager.compute(torch.tensor([0]))
      assert curriculum.current_stage == expected_stage

    assert env.reward_manager.get_term_cfg("place_success").weight == 12.0
    assert (
      env.termination_manager.get_term_cfg("object_lost").params["minimum_height"] > 0.0
    )
  finally:
    env.close()


def test_stationary_grasp_curriculum_restores_scene_after_standing() -> None:
  cfg = talos_tabletop_stationary_grasp_env_cfg(object_shape="cube", play=False)
  cfg.scene.num_envs = 2
  cfg.curriculum["task_stage"].params["evaluation_episodes"] = (2, 4096, 4096, 4096)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    observations, _ = env.reset(seed=5)
    env.sim.forward()
    curriculum = env.curriculum_manager.get_term_cfg("task_stage").func

    assert curriculum.current_stage == 0
    assert torch.all(env.scene["table"].data.root_link_pos_w[:, 0] > 5.0)
    assert torch.all(env.scene["object"].data.root_link_pos_w[:, 0] > 5.0)
    assert torch.count_nonzero(
      observations["actor"][..., 117:117 + MAX_COLLISION_OBSTACLES * 11]
    ) == 0

    env.episode_length_buf[:] = 1
    env.reward_manager._episode_sums["standing_success"][:] = 1.0
    env._reset_idx(torch.arange(2))
    env.sim.forward()

    assert curriculum.current_stage == 1
    assert torch.allclose(
      env.scene["table"].data.root_link_pos_w[:, 2],
      torch.full((2,), 0.82),
      atol=1.0e-5,
    )
    assert torch.all(
      env.scene["object"].data.root_link_pos_w[:, 2] > TABLE_TOP_HEIGHT_M
    )
    assert env.reward_manager.get_term_cfg("approach_object").weight == 4.0
    assert env.reward_manager.get_term_cfg("navigation_progress").weight == 0.0
    assert env.reward_manager.get_term_cfg("place_success").weight == 0.0
  finally:
    env.close()


def test_stationary_face_grasp_rewards_run_without_changing_observations() -> None:
  cfg = talos_tabletop_stationary_grasp_env_cfg(object_shape="cube", play=False)
  cfg.scene.num_envs = 1
  cfg.curriculum["task_stage"].params["initial_stage"] = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    observations, _ = env.reset(seed=7)
    assert observations["actor"].shape == (1, 222)
    assert observations["critic"].shape == (1, 225)
    env.step(torch.zeros((1, env.action_manager.total_action_dim)))
    assert "Metrics/face_contact_area_proxy" in env.extras["log"]
    assert "Metrics/force_closure" in env.extras["log"]
    assert "Metrics/face_grasp_success" in env.extras["log"]
  finally:
    env.close()
