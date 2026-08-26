import torch
from mjlab.envs import ManagerBasedRlEnv
from talos_tabletop.tasks.tabletop.env_cfg import talos_tabletop_grasp_env_cfg


def test_normal_foot_landing_does_not_reset_or_move_object() -> None:
  cfg = talos_tabletop_grasp_env_cfg(object_shape="cube", play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=3)
    obj = env.scene["object"]
    initial_object_xy = obj.data.root_link_pos_w[0, :2].clone()
    zero_action = torch.zeros((1, env.action_manager.total_action_dim))

    for _ in range(8):
      env.step(zero_action)
      assert not bool(env.reset_buf[0])

    assert torch.linalg.vector_norm(
      obj.data.root_link_pos_w[0, :2] - initial_object_xy
    ) < 1.0e-4
    assert bool((env.scene["feet_ground_contact"].data.found[0] > 0).all())
  finally:
    env.close()
