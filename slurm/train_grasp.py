"""Register this source checkout before entering MJLab's training CLI."""

from __future__ import annotations

import os
import runpy

import talos_tabletop.tasks  # noqa: F401  # Populate MJLab's task registry.


def _enable_weights_only_initialization() -> None:
  """Make the regular resume path load networks but start a fresh run."""
  if os.environ.get("TALOS_INIT_WEIGHTS_ONLY") != "1":
    return
  from mjlab.rl import MjlabOnPolicyRunner

  original_load = MjlabOnPolicyRunner.load

  def load_weights_only(self, path, load_cfg=None, strict=True, map_location=None):
    del load_cfg
    print(f"[INFO] Initializing actor/critic weights only from: {path}")
    return original_load(
      self,
      path,
      load_cfg={
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
      },
      strict=strict,
      map_location=map_location,
    )

  MjlabOnPolicyRunner.load = load_weights_only

if __name__ == "__main__":
  _enable_weights_only_initialization()
  runpy.run_module("mjlab.scripts.train", run_name="__main__")
