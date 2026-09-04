from types import SimpleNamespace

import pytest
import torch
from talos_tabletop.tasks.tabletop import mdp
from talos_tabletop.tasks.tabletop.mdp import rewards as reward_functions
from talos_tabletop.tasks.tabletop.mdp.rewards import (
  _object_lift_state,
  _paired_face_and_wrench_quality,
)


class _Scene(dict):
  env_origins: torch.Tensor


def _paired_quality(
  normals: torch.Tensor,
  forces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return _paired_face_and_wrench_quality(
    slot_quality=torch.ones((1, 2)),
    normal_object=normals.unsqueeze(0),
    force_object=forces.unsqueeze(0),
    force_magnitude=torch.linalg.vector_norm(forces, dim=1).unsqueeze(0),
    primary_count=2,
    slots_per_primary=1,
    contacts_for_full_area=2,
    links_for_full_area=2,
    minimum_force=0.25,
    friction_coefficient=1.5,
  )


def test_one_sided_face_contacts_have_zero_grasp_quality() -> None:
  paired_area, wrench_quality, grasp_quality = _paired_quality(
    normals=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    forces=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
  )
  assert paired_area.item() == 0.0
  assert wrench_quality.item() == 0.0
  assert grasp_quality.item() == 0.0


def test_balanced_antipodal_contacts_have_full_grasp_quality() -> None:
  paired_area, wrench_quality, grasp_quality = _paired_quality(
    normals=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    forces=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
  )
  assert paired_area.item() == pytest.approx(1.0)
  assert wrench_quality.item() == pytest.approx(1.0)
  assert grasp_quality.item() == pytest.approx(1.0)


def test_sliding_antipodal_contacts_fail_task_wrench_check() -> None:
  paired_area, wrench_quality, grasp_quality = _paired_quality(
    normals=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    forces=torch.tensor([[0.25, 1.0, 0.0], [-0.25, 1.0, 0.0]]),
  )
  assert paired_area.item() == pytest.approx(1.0)
  assert wrench_quality.item() == 0.0
  assert grasp_quality.item() == 0.0


def test_approach_record_progress_cannot_be_farmed_by_waiting() -> None:
  robot = SimpleNamespace(
    site_names=["grasp"],
    data=SimpleNamespace(site_pos_w=torch.tensor([[[0.0, 0.0, 0.0]]])),
  )
  obj = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=torch.tensor([[1.0, 0.0, 0.0]]))
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    scene=_Scene(robot=robot, object=obj),
    extras={"log": {}},
  )
  term = mdp.site_object_record_progress(cfg=None, env=env)

  assert term(env, site_name="grasp", std=1.0).item() == 0.0
  obj.data.root_link_pos_w[0, 0] = 0.5
  assert term(env, site_name="grasp", std=1.0).item() > 0.0
  assert term(env, site_name="grasp", std=1.0).item() == 0.0


def test_grasp_ready_success_emits_once_after_sustained_quality(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    extras={"log": {}},
  )
  cfg = SimpleNamespace(params={"required_duration_s": 0.04})
  quality = torch.tensor([1.0])
  monkeypatch.setattr(
    reward_functions,
    "_contact_face_metrics",
    lambda *args, **kwargs: (quality, quality, quality, quality),
  )
  term = mdp.sustained_grasp_ready_success(cfg=cfg, env=env)

  params = {
    "sensor_name": "contact",
    "required_duration_s": 0.04,
    "minimum_grasp_quality": 0.2,
  }
  assert term(env, **params).item() == 0.0
  assert term(env, **params).item() == pytest.approx(1.0 / env.step_dt)
  assert term(env, **params).item() == 0.0


def test_cube_tilt_does_not_count_as_airborne_lift() -> None:
  half_sqrt = 2.0**-0.5
  vertical_radius = (0.03**2 + 0.03**2) ** 0.5
  obj = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.86 + vertical_radius]]),
      root_link_quat_w=torch.tensor([[half_sqrt + 1.0, 0.0, half_sqrt, 0.0]])
      / (2.0 + 2.0 * half_sqrt) ** 0.5,
    )
  )

  center_lift, clearance, effective_lift = _object_lift_state(
    obj,
    initial_center_height=0.89,
    table_height=0.86,
    object_half_extents=(0.03, 0.03, 0.03),
    sphere_radius=None,
    clearance_margin=0.0,
  )
  assert center_lift.item() > 0.01
  assert clearance.item() == pytest.approx(0.0, abs=1.0e-6)
  assert effective_lift.item() == pytest.approx(0.0, abs=1.0e-6)

  obj.data.root_link_pos_w[0, 2] += 0.05
  _, clearance, effective_lift = _object_lift_state(
    obj,
    initial_center_height=0.89,
    table_height=0.86,
    object_half_extents=(0.03, 0.03, 0.03),
    sphere_radius=None,
    clearance_margin=0.0,
  )
  assert clearance.item() == pytest.approx(0.05, abs=1.0e-6)
  assert effective_lift.item() == pytest.approx(0.05, abs=1.0e-6)


def test_contact_lift_hold_requires_airborne_height_and_distinct_links(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  obj = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.952]]),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
  )
  env = SimpleNamespace(
    scene=_Scene(object=obj, contact=object()),
    extras={"log": {}},
  )
  enough_links = torch.tensor([True])
  monkeypatch.setattr(
    reward_functions,
    "_contact_link_mask",
    lambda sensor, minimum_links: enough_links,
  )
  params = {
    "initial_center_height": 0.89,
    "table_height": 0.86,
    "target_lift_height": 0.06,
    "sensor_name": "contact",
    "minimum_contact_links": 2,
    "object_half_extents": (0.03, 0.03, 0.03),
  }

  assert mdp.contact_verified_lift_hold(env, **params).item() == pytest.approx(1.0)
  enough_links[0] = False
  assert mdp.contact_verified_lift_hold(env, **params).item() == 0.0
  enough_links[0] = True
  obj.data.root_link_pos_w[0, 2] = 0.89
  assert mdp.contact_verified_lift_hold(env, **params).item() == 0.0


def test_contact_height_target_penalizes_height_error_speed_and_lost_contact(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  obj = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.97]]),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
      root_link_vel_w=torch.zeros((1, 6)),
    )
  )
  env = SimpleNamespace(
    scene=_Scene(object=obj, contact=object()),
    extras={"log": {}},
  )
  env.device = "cpu"
  contact_gate = torch.tensor([True])
  monkeypatch.setattr(
    reward_functions,
    "_contact_link_mask",
    lambda *args, **kwargs: contact_gate,
  )
  params = {
    "initial_center_height": 0.89,
    "table_height": 0.86,
    "target_lift_height": 0.08,
    "height_error_std": 0.025,
    "object_speed_std": 0.08,
    "sensor_name": "contact",
    "minimum_contact_links": 2,
    "object_half_extents": (0.03, 0.03, 0.03),
    "clearance_margin": 0.0,
  }

  at_target = mdp.contact_verified_height_target(env, **params)
  assert at_target.item() == pytest.approx(1.0)
  obj.data.root_link_pos_w[0, 2] = 0.93
  wrong_height = mdp.contact_verified_height_target(env, **params)
  assert 0.0 < wrong_height.item() < at_target.item()
  obj.data.root_link_pos_w[0, 2] = 0.97
  obj.data.root_link_vel_w[0, 0] = 0.20
  moving = mdp.contact_verified_height_target(env, **params)
  assert 0.0 < moving.item() < at_target.item()
  obj.data.root_link_vel_w.zero_()
  contact_gate[:] = False
  assert mdp.contact_verified_height_target(env, **params).item() == 0.0


def test_excessive_height_penalty_starts_above_thirty_centimeters() -> None:
  obj = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 1.19]]))
  )
  env = SimpleNamespace(
    scene=_Scene(object=obj),
    extras={"log": {}},
  )
  assert mdp.excessive_object_lift_height(
    env, initial_center_height=0.89
  ).item() == pytest.approx(0.0)
  obj.data.root_link_pos_w[0, 2] = 1.24
  assert mdp.excessive_object_lift_height(
    env, initial_center_height=0.89
  ).item() == pytest.approx(0.25)


def test_locomotion_phase_distinguishes_tracking_and_manipulation_modes() -> None:
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    step_dt=0.02,
    episode_length_buf=torch.tensor([0, 10]),
  )

  tracking = mdp.locomotion_phase(
    env, period=0.8, control_mode="position_tracking"
  )
  manipulation = mdp.locomotion_phase(
    env, period=0.8, control_mode="manipulation"
  )

  assert tracking.shape == (2, 2)
  assert torch.allclose(torch.linalg.vector_norm(tracking, dim=1), torch.ones(2))
  assert torch.count_nonzero(manipulation) == 0


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

  robot.data.root_link_pos_w[0, 2] = 0.8
  assert mdp.base_target_speed_above_threshold(
    env, minimum_height=0.85, **params
  ).item() == 0.0


def test_navigation_progress_is_gated_by_facing_the_table() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[-2.6, 0.0, 1.0]]),
      root_link_vel_w=torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
  )
  scene = _Scene(robot=robot)
  scene.env_origins = torch.zeros((1, 3))
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene=scene,
    extras={"log": {}},
  )
  term = mdp.base_target_progress(cfg=None, env=env)
  params = {
    "target_position": (0.0, 0.0),
    "heading_target_position": (0.75, 0.0),
    "maximum_speed": 0.25,
    "maximum_projected_gravity_xy": 0.25,
  }

  assert term(env, **params).item() == pytest.approx(0.2)
  half_sqrt = 2.0**-0.5
  robot.data.root_link_quat_w[:] = torch.tensor(
    [[half_sqrt, 0.0, 0.0, half_sqrt]]
  )
  assert term(env, **params).item() == pytest.approx(0.0, abs=1.0e-6)
