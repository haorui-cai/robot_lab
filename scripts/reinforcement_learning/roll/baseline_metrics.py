#!/usr/bin/env python3
"""Compute baseline metrics from heuristic ROLL csv logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass
class EpisodeMetrics:
    file: str
    elapsed_time_s: float
    distance_m: float
    avg_speed_mps: float
    energy_motor_j: float
    specific_cost_j_per_m: float
    power_peak_w: float
    roll_peak_rad: float
    success: bool


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if _is_float(value) else default


def _iter_data_rows(rows: Iterable[dict[str, str]]) -> Iterable[dict[str, str]]:
    for row in rows:
        elapsed = row.get("elapsed_time_system_clock", "")
        if not _is_float(elapsed):
            continue
        yield row


def parse_log(csv_path: Path, distance_axis: str = "x") -> EpisodeMetrics | None:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        data_rows = list(_iter_data_rows(reader))

    if not data_rows:
        return None

    start = data_rows[0]
    end = data_rows[-1]
    elapsed_time = max(_safe_float(end, "elapsed_time_system_clock") - _safe_float(start, "elapsed_time_system_clock"), 1e-6)

    x0 = _safe_float(start, "x")
    y0 = _safe_float(start, "y")
    x1 = _safe_float(end, "x")
    y1 = _safe_float(end, "y")

    if distance_axis == "xy":
        distance = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    elif distance_axis == "y":
        distance = abs(y1 - y0)
    else:
        distance = abs(x1 - x0)

    avg_speed = distance / elapsed_time
    energy_motor = _safe_float(end, "energy_motor")
    specific_cost = energy_motor / max(distance, 1e-6)
    power_peak = max(_safe_float(r, "power_motor") for r in data_rows)
    roll_peak = max(abs(_safe_float(r, "roll")) for r in data_rows)
    success = distance >= 0.6 and roll_peak > 1.0

    return EpisodeMetrics(
        file=csv_path.name,
        elapsed_time_s=elapsed_time,
        distance_m=distance,
        avg_speed_mps=avg_speed,
        energy_motor_j=energy_motor,
        specific_cost_j_per_m=specific_cost,
        power_peak_w=power_peak,
        roll_peak_rad=roll_peak,
        success=success,
    )


def summarize(metrics: list[EpisodeMetrics]) -> dict[str, float | int]:
    return {
        "num_episodes": len(metrics),
        "success_rate": sum(1 for m in metrics if m.success) / max(len(metrics), 1),
        "distance_m_mean": mean(m.distance_m for m in metrics),
        "avg_speed_mps_mean": mean(m.avg_speed_mps for m in metrics),
        "energy_motor_j_mean": mean(m.energy_motor_j for m in metrics),
        "specific_cost_j_per_m_mean": mean(m.specific_cost_j_per_m for m in metrics),
        "power_peak_w_mean": mean(m.power_peak_w for m in metrics),
        "roll_peak_rad_mean": mean(m.roll_peak_rad for m in metrics),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute heuristic ROLL baseline metrics from CSV logs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/eureka/docker-20.04/unitree_guide/src/unitree_guide/data/SimEnergyData"),
        help="Directory containing SimEnergyData CSV files.",
    )
    parser.add_argument(
        "--distance-axis",
        choices=("x", "y", "xy"),
        default="x",
        help="Axis used to evaluate rolling distance.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save aggregated metrics as JSON.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        help="If > 0, only use latest N files.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    csv_files = sorted(args.input_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if args.latest > 0:
        csv_files = csv_files[: args.latest]
    csv_files = list(reversed(csv_files))

    episodes: list[EpisodeMetrics] = []
    for csv_path in csv_files:
        metric = parse_log(csv_path, distance_axis=args.distance_axis)
        if metric is not None:
            episodes.append(metric)

    if not episodes:
        raise RuntimeError(f"No valid data rows found in {args.input_dir}")

    aggregate = summarize(episodes)
    print(json.dumps({"aggregate": aggregate, "episodes": [asdict(e) for e in episodes]}, indent=2, ensure_ascii=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"aggregate": aggregate, "episodes": [asdict(e) for e in episodes]}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
