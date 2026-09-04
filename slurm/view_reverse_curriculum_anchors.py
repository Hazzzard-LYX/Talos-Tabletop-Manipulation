"""Browse reverse-curriculum anchor poses in the MJLab Viser viewer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from html import escape
from pathlib import Path

import talos_tabletop.tasks  # noqa: F401  # Register project tasks.
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from talos_tabletop.tasks.tabletop.env_cfg import (
  talos_tabletop_stable_contact_lift_env_cfg,
)

TASK = "Mjlab-Tabletop-Stable-Contact-Lift-Cube-Talos-v0"
FAMILY_LABELS = {
  "axial": "axial / 轴向",
  "lateral": "lateral / 横向",
  "wrist": "wrist / 手腕",
  "gripper": "gripper / 夹爪",
}
COMPONENT_LABELS = (
  "轴向位置",
  "横向位置",
  "手腕旋转",
  "夹爪开度",
  "关节极限",
  "基座倾斜",
  "基座线速度",
  "基座角速度",
  "手-物相对速度",
)
COMPONENT_COLORS = (
  (70, 150, 255),
  (255, 155, 55),
  (80, 205, 130),
  (230, 105, 170),
  (150, 115, 235),
  (235, 195, 75),
  (70, 190, 195),
  (225, 105, 85),
  (135, 150, 165),
)


def _inject_anchors(env: ManagerBasedRlEnv, bank: dict) -> None:
  anchors = bank["anchors"]
  robot = env.scene["robot"]
  obj = env.scene["object"]
  if tuple(bank["joint_names"]) != tuple(robot.joint_names):
    raise ValueError("Anchor joint ordering does not match the MJLab model.")

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


class AnchorBrowserViewer(ViserPlayViewer):
  """Viser viewer with stage-aware anchor navigation and scene labels."""

  def __init__(self, *args, bank: dict, start_index: int = 0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._bank = bank
    self._anchors = bank["anchors"]
    self._display_ids = self._make_display_ids()
    self._selected_index = max(0, min(start_index, len(self._anchors) - 1))
    self._updating_controls = False
    self.pause()

  def _make_display_ids(self) -> list[str]:
    stage_counts: dict[int, int] = {}
    display_ids = []
    for anchor in self._anchors:
      stage = int(anchor["requested_stage"])
      stage_counts[stage] = stage_counts.get(stage, 0) + 1
      display_ids.append(f"S{stage:02d}-{stage_counts[stage]:02d}")
    return display_ids

  def _option(self, index: int) -> str:
    anchor = self._anchors[index]
    return (
      f"{self._display_ids[index]} | {anchor['family']} | "
      f"D={float(anchor['difficulty']):.3f}"
    )

  def setup(self) -> None:
    super().setup()
    self._scene.show_only_selected = True
    self._scene.camera_tracking_enabled = True
    self._scene.env_idx = self._selected_index
    self._environment_slider = None
    self._hide_others_checkbox = None
    for handle in self._server.gui._gui_input_handle_from_uuid.values():
      label = getattr(handle, "label", None)
      if label == "Select" and getattr(handle, "max", None) == len(self._anchors) - 1:
        self._environment_slider = handle
      elif label == "Hide others":
        self._hide_others_checkbox = handle
    if self._hide_others_checkbox is not None:
      self._hide_others_checkbox.value = True

    with self._server.gui.add_folder("Anchor Browser / 锚点浏览器", expand_by_default=True):
      stages = sorted({int(anchor["requested_stage"]) for anchor in self._anchors})
      stage_options = [f"Stage {stage:02d}" for stage in stages]
      selected_stage = int(self._anchors[self._selected_index]["requested_stage"])
      self._stage_dropdown = self._server.gui.add_dropdown(
        "Stage",
        options=stage_options,
        initial_value=f"Stage {selected_stage:02d}",
      )
      self._anchor_dropdown = self._server.gui.add_dropdown(
        "Anchor",
        options=[self._option(index) for index in range(len(self._anchors))],
        initial_value=self._option(self._selected_index),
      )
      navigation = self._server.gui.add_button_group(
        "Navigate",
        options=["← Previous", "Next →"],
      )
      restore = self._server.gui.add_button("Restore all 100 anchors")
      self._anchor_info = self._server.gui.add_markdown("")

      @self._stage_dropdown.on_update
      def _(_) -> None:
        if self._updating_controls:
          return
        stage = int(self._stage_dropdown.value.split()[-1])
        index = next(
          i
          for i, anchor in enumerate(self._anchors)
          if int(anchor["requested_stage"]) == stage
        )
        self._select_anchor(index)

      @self._anchor_dropdown.on_update
      def _(_) -> None:
        if self._updating_controls:
          return
        self._select_anchor(
          [self._option(index) for index in range(len(self._anchors))].index(
            self._anchor_dropdown.value
          )
        )

      @navigation.on_click
      def _(event) -> None:
        delta = -1 if event.target.value == "← Previous" else 1
        self._select_anchor((self._selected_index + delta) % len(self._anchors))

      @restore.on_click
      def _(_) -> None:
        self.request_reset()

    self._scene_label = self._server.scene.add_label(
      "/anchor-browser/label",
      "",
      position=(0.0, 0.0, 1.25),
      font_size_mode="screen",
      font_screen_scale=1.25,
      depth_test=False,
      anchor="bottom-center",
    )
    self._grasp_marker = self._server.scene.add_icosphere(
      "/anchor-browser/grasp-marker",
      radius=0.035,
      color=COMPONENT_COLORS[0],
      subdivisions=2,
      opacity=0.9,
      cast_shadow=False,
      receive_shadow=False,
    )
    self._select_anchor(self._selected_index)
    self._scene.needs_update = True

  def _select_anchor(self, index: int) -> None:
    self._selected_index = index
    anchor = self._anchors[index]
    stage = int(anchor["requested_stage"])
    self._scene.env_idx = index
    self._scene.needs_update = True
    if self._environment_slider is not None:
      self._environment_slider.value = index

    self._updating_controls = True
    try:
      self._stage_dropdown.value = f"Stage {stage:02d}"
      self._anchor_dropdown.value = self._option(index)
    finally:
      self._updating_controls = False

    components = [float(value) for value in anchor["component_difficulties"]]
    dominant_index = max(range(len(components)), key=components.__getitem__)
    dominant = COMPONENT_LABELS[dominant_index]
    display_id = self._display_ids[index]
    family = FAMILY_LABELS.get(anchor["family"], anchor["family"])
    self._scene_label.text = (
      f"{display_id}  |  Stage {stage:02d}  |  "
      f"{family}  |  D={float(anchor['difficulty']):.3f}"
    )
    self._grasp_marker.color = COMPONENT_COLORS[dominant_index]

    component_lines = "\n".join(
      f"- **{escape(name)}:** `{value:.3f}`"
      for name, value in zip(COMPONENT_LABELS, components, strict=True)
      if value > 0.0
    )
    self._anchor_info.content = (
      f"### {display_id} · Stage {stage:02d}\n"
      f"- **序号:** `{index + 1} / {len(self._anchors)}`\n"
      f"- **扰动族:** {escape(family)}\n"
      f"- **总体难度:** `{float(anchor['difficulty']):.3f}`\n"
      f"- **主导分量:** {escape(dominant)}\n"
      f"- **期望接触链路:** `{int(anchor['expected_contact_link_count'])}`\n\n"
      f"#### 难度分量\n{component_lines}"
    )
    self._update_annotation_positions()

  def _update_annotation_positions(self) -> None:
    env = self.env.unwrapped
    robot = env.scene["robot"]
    site_id = robot.site_names.index("right_grasp_center")
    root_position = robot.data.root_link_pos_w[self._selected_index]
    grasp_position = robot.data.site_pos_w[self._selected_index, site_id]
    local_grasp = (grasp_position - root_position).detach().cpu().numpy()
    self._grasp_marker.position = local_grasp
    self._scene_label.position = (0.0, 0.0, 1.25)

  def sync_env_to_viewer(self) -> None:
    self._update_annotation_positions()
    super().sync_env_to_viewer()

  def reset_environment(self) -> None:
    super().reset_environment()
    with self._sim_lock:
      _inject_anchors(self.env.unwrapped, self._bank)
    self._scene.needs_update = True
    self._select_anchor(self._selected_index)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--anchor-bank", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--port", type=int, default=18081)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--env-spacing", type=float, default=3.0)
  parser.add_argument("--start-index", type=int, default=0)
  parser.add_argument("--seed", type=int, default=20260904)
  args = parser.parse_args()

  if not args.anchor_bank.is_file():
    raise FileNotFoundError(args.anchor_bank)
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)

  bank = torch.load(args.anchor_bank, map_location="cpu", weights_only=False)
  anchors = bank.get("anchors", [])
  if len(anchors) != 100:
    raise ValueError(f"Expected exactly 100 anchors, found {len(anchors)}.")

  configure_torch_backends()
  cfg = talos_tabletop_stable_contact_lift_env_cfg(play=True)
  cfg.scene.num_envs = len(anchors)
  cfg.scene.env_spacing = args.env_spacing
  cfg.viewer.distance = 3.5
  cfg.viewer.azimuth = 100.0
  cfg.viewer.elevation = 8.0
  cfg.seed = args.seed
  agent_cfg = load_rl_cfg(TASK)
  base_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  _inject_anchors(base_env, bank)
  server = viser.ViserServer(
    host="0.0.0.0",
    port=args.port,
    label="TALOS Anchor Browser",
  )
  print(f"ANCHORS={args.anchor_bank}", flush=True)
  print(f"CHECKPOINT={args.checkpoint}", flush=True)
  print(f"ANCHOR_COUNT={len(anchors)}", flush=True)
  print(f"VISER_URL=http://127.0.0.1:{args.port}", flush=True)
  try:
    AnchorBrowserViewer(
      env,
      policy,
      viser_server=server,
      bank=bank,
      start_index=args.start_index,
    ).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
