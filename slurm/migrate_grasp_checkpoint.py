"""Expand tabletop policy observations while preserving learned weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from talos_tabletop.checkpoint_migration import migrate_checkpoint


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--actor-added-dims", type=int, default=2)
  parser.add_argument("--critic-added-dims", type=int, default=2)
  args = parser.parse_args()
  migrate_checkpoint(
    args.input,
    args.output,
    args.actor_added_dims,
    args.critic_added_dims,
  )


if __name__ == "__main__":
  main()
