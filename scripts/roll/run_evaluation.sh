#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 PLAN_JSON OUTPUT_DIR" >&2
  exit 2
fi
roll_eval_python=${ROLL_EVAL_PYTHON:-/home/eureka/miniconda3/envs/isaaclab/bin/python}
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$2"
"$roll_eval_python" -u scripts/roll/evaluate_policy.py --plan "$1" --output "$2" \
  --num-envs 128 --episodes-per-env 2 --max-steps 1010 --headless --device cuda:0 2>&1 | tee "$2/evaluate.log"
if rg -i '\[Error\]|overflow|increase.*capacity|capacity.*exceeded' "$2/evaluate.log"; then
  echo "Simulator errors/capacity warnings found: inspect the log before accepting results." >&2
  exit 1
fi
"$roll_eval_python" scripts/roll/report_evaluation.py --input "$2"
