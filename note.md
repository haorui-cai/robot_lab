
2026-07-27_18-43-28_tqbot2_walk_recovery_v2

cd /home/eureka/robot_lab
source /home/eureka/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
export OMNI_KIT_ACCEPT_EULA=YES

python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-TQBot2-v0 \
  --num_envs 1 \
  --checkpoint /home/eureka/robot_lab/logs/rsl_rl/tqbot2_rough/2026-07-27_18-43-28_tqbot2_walk_recovery_v2/model_19999.pt

  tensorboard \
  --logdir /home/eureka/robot_lab/logs/rsl_rl/tqbot2_rough \
  --port 6006