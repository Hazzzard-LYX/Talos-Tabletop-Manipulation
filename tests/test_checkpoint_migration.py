from pathlib import Path

import torch
from talos_tabletop.checkpoint_migration import migrate_checkpoint


def _model_state(observation_dim: int, output_dim: int) -> dict[str, torch.Tensor]:
  return {
    "obs_normalizer._mean": torch.full((1, observation_dim), 2.0),
    "obs_normalizer._var": torch.full((1, observation_dim), 3.0),
    "obs_normalizer._std": torch.full((1, observation_dim), 4.0),
    "obs_normalizer.count": torch.tensor(5.0),
    "mlp.0.weight": torch.arange(
      4 * observation_dim, dtype=torch.float
    ).reshape(4, observation_dim),
    "mlp.0.bias": torch.zeros(4),
    "mlp.2.weight": torch.zeros(output_dim, 4),
  }


def test_checkpoint_migration_appends_zero_initialized_phase_columns(
  tmp_path: Path,
) -> None:
  source = tmp_path / "source.pt"
  destination = tmp_path / "migrated.pt"
  actor = _model_state(220, 29)
  critic = _model_state(223, 1)
  torch.save(
    {
      "actor_state_dict": actor,
      "critic_state_dict": critic,
      "optimizer_state_dict": {"old": True},
      "iter": 1000,
      "infos": {"env_state": {"common_step_counter": 24024}},
    },
    source,
  )

  migrate_checkpoint(source, destination, actor_added_dims=2, critic_added_dims=2)
  migrated = torch.load(destination, weights_only=False)

  assert migrated["actor_state_dict"]["mlp.0.weight"].shape == (4, 222)
  assert migrated["critic_state_dict"]["mlp.0.weight"].shape == (4, 225)
  assert torch.equal(
    migrated["actor_state_dict"]["mlp.0.weight"][:, :220],
    actor["mlp.0.weight"],
  )
  assert torch.count_nonzero(
    migrated["actor_state_dict"]["mlp.0.weight"][:, 220:]
  ) == 0
  assert torch.equal(
    migrated["actor_state_dict"]["obs_normalizer._mean"][:, 220:],
    torch.zeros((1, 2)),
  )
  assert torch.equal(
    migrated["actor_state_dict"]["obs_normalizer._var"][:, 220:],
    torch.ones((1, 2)),
  )
  assert migrated["iter"] == 0
  assert migrated["infos"] == {}
  assert "optimizer_state_dict" not in migrated


def test_checkpoint_migration_can_strip_training_state_without_expansion(
  tmp_path: Path,
) -> None:
  source = tmp_path / "source.pt"
  destination = tmp_path / "weights_only.pt"
  actor = _model_state(222, 29)
  critic = _model_state(225, 1)
  torch.save(
    {
      "actor_state_dict": actor,
      "critic_state_dict": critic,
      "optimizer_state_dict": {"old": True},
      "iter": 1100,
    },
    source,
  )

  migrate_checkpoint(source, destination, actor_added_dims=0, critic_added_dims=0)
  migrated = torch.load(destination, weights_only=False)

  assert migrated["actor_state_dict"]["mlp.0.weight"].shape == (4, 222)
  assert migrated["critic_state_dict"]["mlp.0.weight"].shape == (4, 225)
  assert migrated["iter"] == 0
  assert "optimizer_state_dict" not in migrated
