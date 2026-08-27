"""Register this source checkout before entering MJLab's training CLI."""

from __future__ import annotations

import runpy

import talos_tabletop.tasks  # noqa: F401  # Populate MJLab's task registry.

if __name__ == "__main__":
  runpy.run_module("mjlab.scripts.train", run_name="__main__")
