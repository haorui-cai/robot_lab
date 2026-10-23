"""Portable fresh-training command builder. Prints a plan unless --execute is supplied."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "free": "RobotLab-Isaac-Velocity-Rough-TQBot2-FreeGaitScratch-v0",
    "v4": "RobotLab-Isaac-Velocity-Rough-TQBot2-RewardV4Scratch-v0",
}


def build_command(args):
    if args.num_envs < 1 or args.iterations < 1 or not re.fullmatch(r"cuda:\d+", args.device):
        raise ValueError("Use positive environment/update counts and a CUDA device")
    name = args.run_name or f"{args.variant}_scratch_{datetime.now():%Y%m%d_%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Run name must be a simple label")
    return [sys.executable, "-u", str(ROOT / "scripts/reinforcement_learning/rsl_rl/train_tqbot2_ablation.py"),
            "--task", TASKS[args.variant], "--headless", "--device", args.device,
            "--num_envs", str(args.num_envs), "--seed", str(args.seed),
            "--max_iterations", str(args.iterations), "--run_name", name,
            "--experiment_name", f"tqbot2_{args.variant}_scratch", "--logger", "tensorboard",
            "--load_run", "__NO_RESUME__", "--checkpoint", "__NO_CHECKPOINT__",
            "agent.resume=false", "agent.check_for_nan=true", "env.commands.base_velocity.debug_vis=false"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=TASKS, default="free")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command = build_command(args)
    print(json.dumps({"mode": "execute" if args.execute else "plan only", "command": command,
                      "initialization": "random, noresume, noHIM", "cwd": str(ROOT)}, indent=2), flush=True)
    if not args.execute:
        return
    subprocess.run([sys.executable, str(ROOT / "scripts/tools/sync_tqbot2_description.py")], check=True, cwd=ROOT)
    gpu = args.device.split(":")[1]
    free = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                                        "--format=csv,noheader,nounits"], text=True).strip())
    required = 8192 if args.num_envs >= 4096 else 6144 if args.num_envs >= 2048 else 4096
    if free < required:
        raise RuntimeError(f"Only{free}MiB free; provisional reserve{required}MiB. Choose fewer environments explicitly.")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "source/robot_lab"), PYTHONDONTWRITEBYTECODE="1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, "2")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
