"""Visualize a TALOS tabletop grasp checkpoint with the Viser web viewer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import talos_tabletop.tasks  # noqa: F401  # Register project tasks.
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task", default="Mjlab-Tabletop-Grasp-Cube-Talos-v0")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--env-spacing", type=float, default=5.0)
  parser.add_argument("--initial-stage", type=int, default=0)
  parser.add_argument("--port", type=int, default=18080)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  if args.num_envs <= 0:
    raise ValueError("num-envs must be positive")
  if args.env_spacing <= 0.0:
    raise ValueError("env-spacing must be positive")

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  if "task_stage" in env_cfg.curriculum:
    env_cfg.curriculum["task_stage"].params["initial_stage"] = args.initial_stage
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.scene.env_spacing = args.env_spacing
  env_cfg.seed = args.seed

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  # The curriculum applies stage-specific event parameters after the environment's
  # initial reset.  Reset once more so a nonzero play stage immediately restores
  # scene entities such as the table and object.
  if args.initial_stage > 0:
    base_env.reset()
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  print(f"CHECKPOINT={args.checkpoint}", flush=True)
  print(f"LIVESTREAM_PORT={args.port}", flush=True)
  print(f"NUM_ENVS={args.num_envs}", flush=True)
  print(f"ENV_SPACING={args.env_spacing}", flush=True)
  server = viser.ViserServer(
    host="0.0.0.0",
    port=args.port,
    label="talos-tabletop-grasp",
  )
  try:
    ViserPlayViewer(env, policy, viser_server=server).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
