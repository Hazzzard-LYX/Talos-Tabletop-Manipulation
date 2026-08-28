"""PPO configurations for TALOS tabletop manipulation."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def talos_tabletop_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.2,
        "std_type": "log",
        "std_range": (0.05, 0.8),
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="talos_tabletop_reaching",
    clip_actions=1.0,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def talos_tabletop_grasp_ppo_runner_cfg(
  object_shape: str,
) -> RslRlOnPolicyRunnerCfg:
  """Return the first privileged-position grasp-policy runner."""
  cfg = talos_tabletop_ppo_runner_cfg()
  cfg.experiment_name = f"talos_tabletop_grasp_{object_shape}_privileged_position"
  cfg.save_interval = 250
  cfg.max_iterations = 20_000
  return cfg


def talos_tabletop_position_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return the position-tracking-only runner configuration."""
  cfg = talos_tabletop_ppo_runner_cfg()
  cfg.experiment_name = "talos_tabletop_position_tracking"
  cfg.save_interval = 250
  cfg.max_iterations = 20_000
  return cfg
