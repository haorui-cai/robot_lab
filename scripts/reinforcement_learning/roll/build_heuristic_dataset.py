#!/usr/bin/env python3
"""Build imitation-learning dataset from heuristic roll logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _float_row(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _valid_row(row: dict[str, str]) -> bool:
    try:
        float(row.get("elapsed_time_system_clock", "nan"))
        return True
    except ValueError:
        return False


def _obs_from_row(row: dict[str, str], last_action: np.ndarray) -> np.ndarray:
    core = np.array(
        [
            _float_row(row, "roll"),
            _float_row(row, "pitch"),
            _float_row(row, "yaw"),
            _float_row(row, "wx"),
            _float_row(row, "wy"),
            _float_row(row, "wz"),
        ],
        dtype=np.float32,
    )
    q = np.array([_float_row(row, f"q{i}") for i in range(12)], dtype=np.float32)
    dq = np.array([_float_row(row, f"dq{i}") for i in range(12)], dtype=np.float32)
    return np.concatenate([core, q, dq, last_action], axis=0)


def build_dataset(input_dir: Path, max_files: int = 0) -> tuple[np.ndarray, np.ndarray]:
    csv_files = sorted(input_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if max_files > 0:
        csv_files = csv_files[:max_files]
    csv_files = list(reversed(csv_files))

    obs_buffer: list[np.ndarray] = []
    act_buffer: list[np.ndarray] = []

    for csv_path in csv_files:
        with csv_path.open("r", newline="") as f:
            rows = [r for r in csv.DictReader(f) if _valid_row(r)]
        if len(rows) < 2:
            continue

        last_action = np.zeros(12, dtype=np.float32)
        for i in range(len(rows) - 1):
            cur, nxt = rows[i], rows[i + 1]
            q_cur = np.array([_float_row(cur, f"q{j}") for j in range(12)], dtype=np.float32)
            q_nxt = np.array([_float_row(nxt, f"q{j}") for j in range(12)], dtype=np.float32)
            action = np.clip(q_nxt - q_cur, -0.6, 0.6)
            obs = _obs_from_row(cur, last_action)
            obs_buffer.append(obs)
            act_buffer.append(action)
            last_action = action

    if not obs_buffer:
        raise RuntimeError(f"No valid transitions found in {input_dir}.")
    return np.stack(obs_buffer), np.stack(act_buffer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BC dataset from heuristic SimEnergyData CSV logs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing heuristic roll csv logs.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--max-files", type=int, default=0, help="If > 0, only use latest N files.")
    args = parser.parse_args()

    obs, actions = build_dataset(args.input_dir, max_files=args.max_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, obs=obs, actions=actions)
    print(f"Saved dataset to {args.output}")
    print(f"obs shape: {obs.shape}, actions shape: {actions.shape}")


if __name__ == "__main__":
    main()
