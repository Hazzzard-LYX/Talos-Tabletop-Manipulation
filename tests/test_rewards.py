from types import SimpleNamespace

import pytest
import torch
from talos_tabletop.tasks.tabletop import mdp


class _Scene(dict):
  env_origins: torch.Tensor


def test_navigation_progress_is_radial_velocity_before_dt_scaling() -> None:
  root_position = torch.tensor([[-2.6, 0.0, 1.0]])
  robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_position))
  scene = _Scene(robot=robot)
  scene.env_origins = torch.zeros((1, 3))
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    scene=scene,
  )
  term = mdp.base_target_progress(cfg=None, env=env)

  assert term(env, target_position=(0.0, 0.0)).item() == 0.0
  root_position[0, 0] += 0.02

  assert term(env, target_position=(0.0, 0.0)).item() == pytest.approx(1.0)
