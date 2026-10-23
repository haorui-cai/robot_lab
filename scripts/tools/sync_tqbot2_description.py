# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Generate the Isaac model from the pinned private description repository."""
import argparse
from pathlib import Path

from tqbot_description_dependency import DEPENDENCY, rebuild


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Compatibility option; only the pinned canonical package is accepted")
    args = parser.parse_args()
    canonical = DEPENDENCY / "TQBot2/TQBot2_description"
    if args.source is not None and args.source.resolve() != canonical.resolve():
        parser.error("Physical source is now the pinned private repository; edit its Xacro/profile and update the submodule pin")
    print(f"Generated private model: {rebuild()}")


if __name__ == "__main__":
    main()
