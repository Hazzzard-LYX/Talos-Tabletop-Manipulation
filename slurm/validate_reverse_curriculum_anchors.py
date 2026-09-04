"""Validate IK anchor candidates with the learned stable-standing policy."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import talos_tabletop.tasks  # noqa: F401  # Register project tasks.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends
from talos_tabletop.tasks.tabletop.env_cfg import (
  talos_tabletop_stable_contact_lift_env_cfg,
)
from talos_tabletop.tasks.tabletop.mdp.reverse_curriculum import (
  GRASP_DIFFICULTY_COMPONENT_NAMES,
  GraspStateFeatures,
  classify_grasp_state,
  talos_gripper_open_fraction,
)

TASK = "Mjlab-Tabletop-Stable-Contact-Lift-Cube-Talos-v0"
RIGHT_GRIPPER_RANGE_RAD = 0.959931
IK_JOINT_NAMES = (
  "torso_1_joint",
  "torso_2_joint",
  *(f"arm_right_{index}_joint" for index in range(1, 8)),
)


def _initial_classification(
  env: ManagerBasedRlEnv,
  bank: dict,
) -> tuple[torch.Tensor, object]:
  robot = env.scene["robot"]
  obj = env.scene["object"]
  contact = env.scene["right_gripper_object_contact"]
  site_id = robot.site_names.index("right_grasp_center")
  gripper_id = robot.joint_names.index("gripper_right_joint")
  ik_joint_ids = torch.tensor(
    [robot.joint_names.index(name) for name in IK_JOINT_NAMES],
    device=env.device,
  )

  state_dtype = robot.data.site_pos_w.dtype
  reference_position = bank["reference_grasp_position_w"].to(
    device=env.device, dtype=state_dtype
  )
  reference_position = reference_position.unsqueeze(0) + env.scene.env_origins
  reference_quat = (
    bank["reference_grasp_quat_w"]
    .to(device=env.device, dtype=state_dtype)
    .expand(env.num_envs, -1)
  )
  approach_axis = quat_apply(
    reference_quat,
    torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1),
  )
  delta = robot.data.site_pos_w[:, site_id] - reference_position
  signed_axial = (delta * approach_axis).sum(dim=-1)
  lateral = torch.linalg.vector_norm(
    delta - signed_axial.unsqueeze(-1) * approach_axis, dim=-1
  )

  limits = robot.data.joint_pos_limits[:, ik_joint_ids]
  positions = robot.data.joint_pos[:, ik_joint_ids]
  ranges = limits[..., 1] - limits[..., 0]
  normalized_margin = torch.minimum(
    positions - limits[..., 0], limits[..., 1] - positions
  ) / ranges
  joint_limit_risk = ((0.10 - normalized_margin.amin(dim=-1)) / 0.10).clamp(
    0.0, 1.0
  )

  if contact.data.found is None:
    raise RuntimeError("The hand-object contact sensor did not expose found data.")
  primary_count = len(contact.primary_names)
  contact_links = (
    contact.data.found.reshape(env.num_envs, primary_count, -1)
    .gt(0)
    .any(dim=-1)
    .sum(dim=-1)
  )
  relative_speed = torch.linalg.vector_norm(
    robot.data.site_lin_vel_w[:, site_id] - obj.data.root_link_lin_vel_w,
    dim=-1,
  )
  classification = classify_grasp_state(
    GraspStateFeatures(
      axial_position_error_m=signed_axial.abs(),
      lateral_position_error_m=lateral,
      wrist_rotation_error_rad=quat_error_magnitude(
        reference_quat, robot.data.site_quat_w[:, site_id]
      ),
      gripper_open_fraction=talos_gripper_open_fraction(
        robot.data.joint_pos[:, gripper_id], RIGHT_GRIPPER_RANGE_RAD
      ),
      joint_limit_risk=joint_limit_risk,
      projected_gravity_xy=torch.linalg.vector_norm(
        robot.data.projected_gravity_b[:, :2], dim=-1
      ),
      base_linear_speed_mps=torch.linalg.vector_norm(
        robot.data.root_link_lin_vel_w, dim=-1
      ),
      base_angular_speed_rad_s=torch.linalg.vector_norm(
        robot.data.root_link_ang_vel_w, dim=-1
      ),
      hand_object_relative_speed_mps=relative_speed,
      contact_link_count=contact_links,
      verified_grasp=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
  )
  return contact_links, classification


def _inject_candidates(env: ManagerBasedRlEnv, bank: dict) -> None:
  anchors = bank["anchors"]
  robot = env.scene["robot"]
  obj = env.scene["object"]
  if tuple(bank["joint_names"]) != tuple(robot.joint_names):
    raise ValueError("Anchor joint ordering does not match the composed MJLab model.")

  robot_root_state = torch.stack(
    [anchor["robot_root_state"] for anchor in anchors]
  ).to(env.device)
  robot_root_state[:, :3] += env.scene.env_origins
  joint_position = torch.stack(
    [anchor["robot_joint_position"] for anchor in anchors]
  ).to(env.device)
  joint_velocity = torch.stack(
    [anchor["robot_joint_velocity"] for anchor in anchors]
  ).to(env.device)
  object_root_state = torch.stack(
    [anchor["object_root_state"] for anchor in anchors]
  ).to(env.device)
  object_root_state[:, :3] += env.scene.env_origins

  robot.write_root_state_to_sim(robot_root_state)
  robot.write_joint_state_to_sim(joint_position, joint_velocity)
  obj.write_root_state_to_sim(object_root_state)
  env.episode_length_buf.zero_()
  env.action_manager.reset()
  env.observation_manager.reset()
  env.reward_manager.reset()
  env.termination_manager.reset()
  env.sim.forward()


def _standing_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  feet = env.scene["feet_ground_contact"]
  if feet.data.found is None:
    raise RuntimeError("The feet contact sensor did not expose found data.")
  both_feet = feet.data.found.gt(0).all(dim=-1)
  return (
    both_feet
    & (
      torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=-1)
      <= 0.15
    )
    & (
      torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=-1) <= 0.05
    )
    & (
      torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=-1) <= 0.10
    )
  )


def _select_diverse(anchors: list[dict], count: int) -> list[dict]:
  if len(anchors) < count:
    raise RuntimeError(f"Only {len(anchors)} validated anchors are available for {count} slots.")
  descriptors = torch.stack([anchor["diversity_descriptor"] for anchor in anchors])
  selected = [int(torch.linalg.vector_norm(descriptors, dim=-1).argmax().item())]
  while len(selected) < count:
    remaining = [index for index in range(len(anchors)) if index not in selected]
    distances = torch.linalg.vector_norm(
      descriptors[remaining, None, :] - descriptors[selected, :], dim=-1
    )
    selected.append(remaining[int(distances.amin(dim=1).argmax().item())])
  return [anchors[index] for index in selected]


def validate(
  *,
  input_path: Path,
  output_path: Path,
  diagnostic_output_path: Path | None,
  checkpoint: Path,
  anchors_per_stage: int,
  hold_duration_s: float,
  recovery_duration_s: float,
  device: str,
  seed: int,
  allow_incomplete: bool,
) -> None:
  bank = torch.load(input_path, map_location="cpu", weights_only=False)
  anchors = bank["anchors"]
  if not anchors:
    raise ValueError("Candidate bank is empty.")

  configure_torch_backends()
  cfg = talos_tabletop_stable_contact_lift_env_cfg(play=True)
  cfg.scene.num_envs = len(anchors)
  cfg.scene.env_spacing = 3.0
  cfg.seed = seed
  agent_cfg = load_rl_cfg(TASK)
  base_env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  try:
    _inject_candidates(base_env, bank)
    contact_links, classification = _initial_classification(base_env, bank)
    requested_stage = torch.tensor(
      [anchor["requested_stage"] for anchor in anchors], device=device
    )
    initial_stage_matches = classification.stage == requested_stage
    for stage in sorted(set(requested_stage.tolist())):
      stage_mask = requested_stage == stage
      actual_stages, actual_counts = torch.unique(
        classification.stage[stage_mask], return_counts=True
      )
      stage_histogram = ", ".join(
        f"{int(actual)}:{int(count)}"
        for actual, count in zip(actual_stages.tolist(), actual_counts.tolist(), strict=True)
      )
      matching_contacts = contact_links[stage_mask & initial_stage_matches]
      print(
        f"Stage {stage:02d} initial classification [{stage_histogram}]; "
        f"matching contacts={matching_contacts.tolist()}",
        flush=True,
      )
    ever_terminated = torch.zeros(len(anchors), dtype=torch.bool, device=device)
    first_termination_step = torch.full(
      (len(anchors),), -1, dtype=torch.long, device=device
    )
    termination_hits = {
      name: torch.zeros(len(anchors), dtype=torch.bool, device=device)
      for name in base_env.termination_manager.active_terms
    }
    consecutive_stable_steps = torch.zeros(
      len(anchors), dtype=torch.long, device=device
    )
    required_steps = math.ceil(hold_duration_s / base_env.step_dt)
    total_steps = math.ceil(
      (hold_duration_s + recovery_duration_s) / base_env.step_dt
    )
    observations = env.get_observations()
    for step in range(total_steps):
      with torch.inference_mode():
        actions = policy(observations)
      observations, _, dones, _ = env.step(actions)
      dones = dones.bool()
      newly_terminated = dones & ~ever_terminated
      first_termination_step = torch.where(
        newly_terminated,
        torch.full_like(first_termination_step, step + 1),
        first_termination_step,
      )
      ever_terminated |= dones
      for name, hits in termination_hits.items():
        hits |= base_env.termination_manager.get_term(name)
      stable = _standing_mask(base_env) & ~ever_terminated
      consecutive_stable_steps = torch.where(
        stable, consecutive_stable_steps + 1, torch.zeros_like(consecutive_stable_steps)
      )

    standing_passed = consecutive_stable_steps >= required_steps
    accepted = initial_stage_matches & standing_passed & ~ever_terminated
    for index, anchor in enumerate(anchors):
      actual_components = classification.component_difficulties[index].detach().cpu()
      anchor["standing_validation"] = {
        "checkpoint": str(checkpoint),
        "hold_duration_s": hold_duration_s,
        "recovery_duration_s": recovery_duration_s,
        "initial_stage": int(classification.stage[index].item()),
        "initial_difficulty": float(classification.difficulty[index].item()),
        "initial_component_difficulties": actual_components,
        "initial_dominant_component": GRASP_DIFFICULTY_COMPONENT_NAMES[
          int(classification.dominant_component[index].item())
        ],
        "initial_contact_link_count": int(contact_links[index].item()),
        "ever_terminated": bool(ever_terminated[index].item()),
        "first_termination_s": (
          float(first_termination_step[index].item() * base_env.step_dt)
          if first_termination_step[index].item() >= 0
          else None
        ),
        "termination_reasons": tuple(
          name for name, hits in termination_hits.items() if hits[index].item()
        ),
        "final_consecutive_stable_s": float(
          consecutive_stable_steps[index].item() * base_env.step_dt
        ),
        "accepted": bool(accepted[index].item()),
      }

    if diagnostic_output_path is not None:
      diagnostic_bank = dict(bank)
      diagnostic_bank["kind"] = "reverse_curriculum_anchor_validation_diagnostics"
      diagnostic_bank["standing_policy_checkpoint"] = str(checkpoint)
      diagnostic_bank["anchors"] = anchors
      diagnostic_output_path.parent.mkdir(parents=True, exist_ok=True)
      torch.save(diagnostic_bank, diagnostic_output_path)
      print(
        f"Wrote validation diagnostics for {len(anchors)} candidates to "
        f"{diagnostic_output_path}",
        flush=True,
      )

    final_anchors = []
    validation_shortfalls = {}
    requested_stages = sorted({anchor["requested_stage"] for anchor in anchors})
    for stage in requested_stages:
      stage_anchors = [
        anchor for anchor in anchors if anchor["requested_stage"] == stage
      ]
      valid = [
        anchor
        for anchor in stage_anchors
        if anchor["standing_validation"]["accepted"]
      ]
      classification_matches = sum(
        anchor["standing_validation"]["initial_stage"] == stage
        for anchor in stage_anchors
      )
      non_terminated = sum(
        not anchor["standing_validation"]["ever_terminated"]
        for anchor in stage_anchors
      )
      standing_passed_count = sum(
        anchor["standing_validation"]["final_consecutive_stable_s"]
        >= hold_duration_s
        for anchor in stage_anchors
      )
      print(
        f"Stage {stage:02d}: {len(valid)} of "
        f"{len(stage_anchors)} candidates accepted "
        f"(classification={classification_matches}, "
        f"non_terminated={non_terminated}, standing={standing_passed_count})",
        flush=True,
      )
      if len(valid) < anchors_per_stage:
        validation_shortfalls[stage] = {
          "available": len(valid),
          "required": anchors_per_stage,
        }
        if not allow_incomplete:
          raise RuntimeError(
            f"Stage {stage} has only {len(valid)} stable, correctly classified "
            f"anchors; {anchors_per_stage} are required."
          )
      retained_count = min(len(valid), anchors_per_stage)
      chosen = _select_diverse(valid, retained_count) if retained_count else []
      for anchor_index, anchor in enumerate(chosen):
        anchor["anchor_id"] = f"stage-{stage:02d}-anchor-{anchor_index:02d}"
        anchor["stage"] = stage
        anchor.pop("provisional_stage", None)
      final_anchors.extend(chosen)
      print(f"Stage {stage:02d}: retained {len(chosen)} diverse anchors", flush=True)

    bank["kind"] = "reverse_curriculum_validated_ik_anchor_bank"
    bank["standing_policy_checkpoint"] = str(checkpoint)
    bank["anchors_per_stage"] = anchors_per_stage
    bank["validation_shortfalls"] = validation_shortfalls
    bank["anchors"] = final_anchors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output_path)
    print(f"Wrote {len(final_anchors)} validated anchors to {output_path}")
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--diagnostic-output", type=Path)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--anchors-per-stage", type=int, default=10)
  parser.add_argument("--hold-duration-s", type=float, default=5.0)
  parser.add_argument("--recovery-duration-s", type=float, default=2.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=20260903)
  parser.add_argument("--allow-incomplete", action="store_true")
  args = parser.parse_args()
  if not args.input.is_file():
    raise FileNotFoundError(args.input)
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  if args.anchors_per_stage <= 0:
    raise ValueError("anchors-per-stage must be positive.")
  if args.hold_duration_s <= 0.0 or args.recovery_duration_s < 0.0:
    raise ValueError("Validation durations are invalid.")
  validate(
    input_path=args.input,
    output_path=args.output,
    diagnostic_output_path=args.diagnostic_output,
    checkpoint=args.checkpoint,
    anchors_per_stage=args.anchors_per_stage,
    hold_duration_s=args.hold_duration_s,
    recovery_duration_s=args.recovery_duration_s,
    device=args.device,
    seed=args.seed,
    allow_incomplete=args.allow_incomplete,
  )


if __name__ == "__main__":
  main()
