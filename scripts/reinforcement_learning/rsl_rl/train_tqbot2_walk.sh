#!/usr/bin/env bash
# Current repository profile; historical snapshot launch recipes are not used.
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${task_repo}/source/robot_lab${PYTHONPATH:+:${PYTHONPATH}}"
cd "${task_repo}"
exec "${TQBOT2_TRAIN_PYTHON:-python}" scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0 \
  --headless \
  agent.algorithm.learning_rate=0.001 \
  agent.algorithm.schedule=adaptive \
  agent.algorithm.entropy_coef=0.01 \
  "$@"
