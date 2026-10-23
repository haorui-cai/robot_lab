#!/usr/bin/env python3
"""Extract stable continuous-roll samples from unitree_guide simulation logs.

The controller cycles through RETRACT (20), LEFT_DEPLOY (25), and
LEFT_SUPPORT (30) while rolling continuously.  Startup and braking phases are
excluded; samples must also reach a configurable share of target speed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf",
)
CONTINUOUS_STAGES = (20, 25, 30)


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"missing {key}")
    return float(value)


def extract(paths: list[Path], min_speed_ratio: float) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    samples: dict[str, list[object]] = {
        "joint_pos": [], "command_pos": [], "joint_vel": [],
        "time_s": [], "stage": [], "source_index": [],
    }
    source_paths: list[str] = []

    for source_index, path in enumerate(paths):
        selected = 0
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    stage = int(number(row, "roll_control_stage"))
                    target_speed = number(row, "roll_target_speed")
                    measured_speed = number(row, "gazebo_xy_speed")
                    braking = int(number(row, "roll_brake_requested"))
                    if (
                        stage not in CONTINUOUS_STAGES
                        or braking != 0
                        or abs(target_speed) < 1.0e-6
                        or abs(measured_speed) < min_speed_ratio * abs(target_speed)
                    ):
                        continue
                    joint_pos = [number(row, f"q{i}") for i in range(12)]
                    command_pos = [number(row, f"cmd_q{i}") for i in range(12)]
                    joint_vel = [number(row, f"dq{i}") for i in range(12)]
                except (TypeError, ValueError):
                    continue

                samples["joint_pos"].append(joint_pos)
                samples["command_pos"].append(command_pos)
                samples["joint_vel"].append(joint_vel)
                samples["time_s"].append(number(row, "elapsed_time_system_clock"))
                samples["stage"].append(stage)
                samples["source_index"].append(source_index)
                selected += 1
        if selected:
            source_paths.append(str(path))

    arrays = {
        "joint_pos": np.asarray(samples["joint_pos"], dtype=np.float32),
        "command_pos": np.asarray(samples["command_pos"], dtype=np.float32),
        "joint_vel": np.asarray(samples["joint_vel"], dtype=np.float32),
        "time_s": np.asarray(samples["time_s"], dtype=np.float64),
        "stage": np.asarray(samples["stage"], dtype=np.int16),
        "source_index": np.asarray(samples["source_index"], dtype=np.int16),
    }
    if not len(arrays["joint_pos"]):
        raise RuntimeError("No stable continuous-roll samples were found.")

    summary: dict[str, object] = {
        "selection": {
            "stages": list(CONTINUOUS_STAGES),
            "brake_requested": 0,
            "min_speed_ratio": min_speed_ratio,
        },
        "sources": source_paths,
        "sample_count": int(len(arrays["joint_pos"])),
        "joint_names": list(JOINT_NAMES),
        "command_position_quantiles": {
            str(percentile): np.quantile(arrays["command_pos"], percentile / 100, axis=0).round(6).tolist()
            for percentile in (1, 5, 50, 95, 99)
        },
        "actual_position_quantiles": {
            str(percentile): np.quantile(arrays["joint_pos"], percentile / 100, axis=0).round(6).tolist()
            for percentile in (1, 5, 50, 95, 99)
        },
    }
    return arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True, help="Quoted glob for controller.csv inputs.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file.")
    parser.add_argument("--min-speed-ratio", type=float, default=0.8)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise SystemExit(f"No files match {args.input_glob!r}.")
    arrays, summary = extract(paths, args.min_speed_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {summary['sample_count']} samples from {len(summary['sources'])} logs to {args.output}")


if __name__ == "__main__":
    main()
