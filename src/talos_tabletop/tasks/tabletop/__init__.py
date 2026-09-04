"""Register TALOS tabletop reaching and grasping tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import (
  talos_tabletop_grasp_env_cfg,
  talos_tabletop_position_tracking_env_cfg,
  talos_tabletop_reaching_env_cfg,
  talos_tabletop_stable_contact_lift_env_cfg,
  talos_tabletop_stationary_grasp_env_cfg,
)
from .rl_cfg import (
  talos_tabletop_grasp_ppo_runner_cfg,
  talos_tabletop_position_tracking_ppo_runner_cfg,
  talos_tabletop_ppo_runner_cfg,
  talos_tabletop_stable_contact_lift_ppo_runner_cfg,
  talos_tabletop_stationary_grasp_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tabletop-Reaching-Talos-v0",
  env_cfg=talos_tabletop_reaching_env_cfg(),
  play_env_cfg=talos_tabletop_reaching_env_cfg(play=True),
  rl_cfg=talos_tabletop_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tabletop-Stable-Contact-Lift-Cube-Talos-v0",
  env_cfg=talos_tabletop_stable_contact_lift_env_cfg(),
  play_env_cfg=talos_tabletop_stable_contact_lift_env_cfg(play=True),
  rl_cfg=talos_tabletop_stable_contact_lift_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tabletop-Position-Tracking-Talos-v0",
  env_cfg=talos_tabletop_position_tracking_env_cfg(),
  play_env_cfg=talos_tabletop_position_tracking_env_cfg(play=True),
  rl_cfg=talos_tabletop_position_tracking_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

for object_shape, label in (("cube", "Cube"), ("sphere", "Sphere")):
  register_mjlab_task(
    task_id=f"Mjlab-Tabletop-Grasp-{label}-Talos-v0",
    env_cfg=talos_tabletop_grasp_env_cfg(object_shape=object_shape),
    play_env_cfg=talos_tabletop_grasp_env_cfg(object_shape=object_shape, play=True),
    rl_cfg=talos_tabletop_grasp_ppo_runner_cfg(object_shape),
    runner_cls=VelocityOnPolicyRunner,
  )
  register_mjlab_task(
    task_id=f"Mjlab-Tabletop-Stationary-Grasp-{label}-Talos-v0",
    env_cfg=talos_tabletop_stationary_grasp_env_cfg(object_shape=object_shape),
    play_env_cfg=talos_tabletop_stationary_grasp_env_cfg(
      object_shape=object_shape, play=True
    ),
    rl_cfg=talos_tabletop_stationary_grasp_ppo_runner_cfg(object_shape),
    runner_cls=VelocityOnPolicyRunner,
  )
