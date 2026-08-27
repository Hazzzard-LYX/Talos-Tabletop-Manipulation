"""Performance-gated curricula for the TALOS tabletop task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class performance_stage_curriculum:
  """Advance a global task stage from completed-episode reward success.

  Each non-final stage names a sparse success reward.  Episodes count as
  successful when that reward accumulated a positive value.  Promotion is
  evaluated over fixed episode windows, making advancement dependent on task
  performance instead of optimizer iterations or environment steps.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._env = env
    self._stage_reward_weights: tuple[dict[str, float], ...] = params[
      "stage_reward_weights"
    ]
    self._promotion_reward_names: tuple[str, ...] = params["promotion_reward_names"]
    self._promotion_success_rates: tuple[float, ...] = params["promotion_success_rates"]
    self._evaluation_episodes: tuple[int, ...] = params["evaluation_episodes"]
    self._stage_termination_params: tuple[dict[str, dict[str, Any]], ...] = params.get(
      "stage_termination_params", tuple({} for _ in self._stage_reward_weights)
    )
    self.current_stage = int(params.get("initial_stage", 0))
    self._window_episodes = 0
    self._window_successes = 0
    self._last_success_rate = 0.0

    if not self._stage_reward_weights:
      raise ValueError("stage_reward_weights must contain at least one stage.")
    if not 0 <= self.current_stage < len(self._stage_reward_weights):
      raise ValueError("initial_stage is outside the configured stage range.")
    expected_promotions = len(self._stage_reward_weights) - 1
    if not (
      len(self._promotion_reward_names)
      == len(self._promotion_success_rates)
      == len(self._evaluation_episodes)
      == expected_promotions
    ):
      raise ValueError(
        "Promotion criteria must be specified for every non-final stage."
      )
    if len(self._stage_termination_params) != len(self._stage_reward_weights):
      raise ValueError("stage_termination_params must match the number of stages.")

    for stage_weights in self._stage_reward_weights:
      for reward_name in stage_weights:
        env.reward_manager.get_term_cfg(reward_name)
    for reward_name in self._promotion_reward_names:
      env.reward_manager.get_term_cfg(reward_name)

    self._apply_stage()

  def _apply_stage(self) -> None:
    for reward_name, weight in self._stage_reward_weights[self.current_stage].items():
      self._env.reward_manager.get_term_cfg(reward_name).weight = weight
    for termination_name, params in self._stage_termination_params[
      self.current_stage
    ].items():
      term_cfg = self._env.termination_manager.get_term_cfg(termination_name)
      term_cfg.params.update(params)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stage_reward_weights: tuple[dict[str, float], ...],
    promotion_reward_names: tuple[str, ...],
    promotion_success_rates: tuple[float, ...],
    evaluation_episodes: tuple[int, ...],
    stage_termination_params: tuple[dict[str, dict[str, Any]], ...],
    initial_stage: int = 0,
  ) -> dict[str, torch.Tensor]:
    del (
      stage_reward_weights,
      promotion_reward_names,
      promotion_success_rates,
      evaluation_episodes,
      stage_termination_params,
      initial_stage,
    )

    if self.current_stage < len(self._promotion_reward_names):
      completed = env.episode_length_buf[env_ids] > 0
      completed_ids = env_ids[completed]
      if len(completed_ids) > 0:
        reward_name = self._promotion_reward_names[self.current_stage]
        episode_rewards = env.reward_manager._episode_sums[reward_name][completed_ids]
        self._window_episodes += len(completed_ids)
        self._window_successes += int((episode_rewards > 0.0).sum().item())

      required_episodes = self._evaluation_episodes[self.current_stage]
      if self._window_episodes >= required_episodes:
        self._last_success_rate = self._window_successes / self._window_episodes
        required_rate = self._promotion_success_rates[self.current_stage]
        if self._last_success_rate >= required_rate:
          self.current_stage += 1
          self._apply_stage()
        self._window_episodes = 0
        self._window_successes = 0

    return {
      "stage": torch.tensor(float(self.current_stage), device=env.device),
      "success_rate": torch.tensor(self._last_success_rate, device=env.device),
      "window_episodes": torch.tensor(float(self._window_episodes), device=env.device),
    }
