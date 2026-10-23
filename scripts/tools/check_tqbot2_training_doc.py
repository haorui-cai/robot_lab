#!/usr/bin/env python3
"""Read-only drift check for the manually audited TQBot2 training document."""

import argparse
import hashlib
import json
from pathlib import Path


def check_files(root, entries):
    failures = []
    for relative, expected in entries.items():
        path = root / relative
        if not path.is_file():
            failures.append((relative, "MISSING"))
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            failures.append((relative, "CHANGED"))
    return failures


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=root / "docs/TQBot2_Current_Training_Config.sources.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    failures = check_files(root, manifest["sha256"])
    for path, reason in failures:
        print(f"{reason}: {path}")
    if failures:
        print("DRIFT: re-audit the training document before updating its hash manifest.")
        return 1
    print(f"MATCH: {len(manifest['sha256'])} files match audit {manifest['audit_date']}.")
    print("This checks file identity, not semantic correctness or full runtime reproducibility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
