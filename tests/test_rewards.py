from types import SimpleNamespace

import pytest
import torch
from talos_tabletop.tasks.tabletop import mdp


class _Scene(dict):
  env_origins: torch.Tensor


def test_navigation_progress_uses_measured_radial_velocity() -> None:
  root_position = torch.tensor([[-2.6, 0.0, 1.0]])
  root_velocity = torch.zeros((1, 6))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=root_position,
      root_link_vel_w=root_velocity,
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
  )
  scene = _Scene(robot=robot)
  scene.env_origins = torch.zeros((1, 3))
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    scene=scene,
    extras={"log": {}},
  )
  term = mdp.base_target_progress(cfg=None, env=env)

  params = {
    "target_position": (0.0, 0.0),
    "maximum_speed": 0.5,
    "maximum_projected_gravity_xy": 0.7,
  }
  assert term(env, **params).item() == 0.0
  root_velocity[0, 0] = 0.3

  assert term(env, **params).item() == pytest.approx(0.3, abs=1.0e-5)
  assert env.extras["log"]["Metrics/base_lin_vel_x_mps"].item() == pytest.approx(
    0.3
  )
  assert env.extras["log"]["Metrics/base_lin_vel_y_mps"].item() == 0.0
  assert env.extras["log"]["Metrics/base_planar_speed_mps"].item() == pytest.approx(
    0.3
  )

  robot.data.projected_gravity_b[0, 0] = 0.7

  assert term(env, **params).item() == 0.0


def test_navigation_speed_has_zero_reward_below_deadband() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[-2.6, 0.0, 1.0]]),
      root_link_vel_w=torch.tensor([[0.049, 0.0, 0.0, 0.0, 0.0, 0.0]]),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
  )
  scene = _Scene(robot=robot)
  scene.env_origins = torch.zeros((1, 3))
  env = SimpleNamespace(num_envs=1, device="cpu", scene=scene)
  params = {
    "target_position": (0.0, 0.0),
    "minimum_speed": 0.05,
    "target_speed": 0.25,
    "maximum_projected_gravity_xy": 0.7,
  }

  assert mdp.base_target_speed_above_threshold(env, **params).item() == 0.0
  robot.data.root_link_vel_w[0, 0] = 0.15
  assert mdp.base_target_speed_above_threshold(
    env, **params
  ).item() == pytest.approx(0.5)
  robot.data.root_link_vel_w[0, 0] = 0.25
  assert mdp.base_target_speed_above_threshold(env, **params).item() == 1.0
