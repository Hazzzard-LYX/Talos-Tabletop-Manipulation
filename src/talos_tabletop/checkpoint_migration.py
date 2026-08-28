"""Checkpoint migrations for TALOS tabletop policies."""

from __future__ import annotations

from pathlib import Path

import torch


def _expand_observation_state(
  state_dict: dict[str, torch.Tensor], added_dims: int
) -> dict[str, torch.Tensor]:
  if added_dims <= 0:
    raise ValueError("added_dims must be positive")
  migrated = dict(state_dict)
  first_layer_key = "mlp.0.weight"
  if first_layer_key not in migrated:
    raise KeyError(f"Checkpoint is missing {first_layer_key}")
  first_layer = migrated[first_layer_key]
  migrated[first_layer_key] = torch.cat(
    (
      first_layer,
      torch.zeros(
        (first_layer.shape[0], added_dims),
        dtype=first_layer.dtype,
        device=first_layer.device,
      ),
    ),
    dim=1,
  )

  for name in ("_mean", "_var", "_std"):
    key = f"obs_normalizer.{name}"
    if key not in migrated:
      continue
    value = migrated[key]
    fill = 0.0 if name == "_mean" else 1.0
    migrated[key] = torch.cat(
      (
        value,
        torch.full(
          (*value.shape[:-1], added_dims),
          fill,
          dtype=value.dtype,
          device=value.device,
        ),
      ),
      dim=-1,
    )
  return migrated


def migrate_checkpoint(
  source: Path,
  destination: Path,
  actor_added_dims: int,
  critic_added_dims: int,
) -> None:
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  if "actor_state_dict" not in checkpoint or "critic_state_dict" not in checkpoint:
    raise KeyError("Expected an rsl-rl actor/critic checkpoint.")

  actor_old_dim = checkpoint["actor_state_dict"]["mlp.0.weight"].shape[1]
  critic_old_dim = checkpoint["critic_state_dict"]["mlp.0.weight"].shape[1]
  migrated = {
    "actor_state_dict": _expand_observation_state(
      checkpoint["actor_state_dict"], actor_added_dims
    ),
    "critic_state_dict": _expand_observation_state(
      checkpoint["critic_state_dict"], critic_added_dims
    ),
    # Weight initialization is deliberately a fresh training run: do not
    # restore optimizer momentum, iteration count, or curriculum counters.
    "iter": 0,
    "infos": {},
  }
  destination.parent.mkdir(parents=True, exist_ok=True)
  torch.save(migrated, destination)
  print(
    "Migrated observations: "
    f"actor {actor_old_dim}->{actor_old_dim + actor_added_dims}, "
    f"critic {critic_old_dim}->{critic_old_dim + critic_added_dims}"
  )
  print(f"Wrote weights-only checkpoint: {destination}")
