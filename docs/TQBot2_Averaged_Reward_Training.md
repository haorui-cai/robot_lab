# TQBot2 两套历史奖励等权平均实验

2026-09-29，用户要求直接将上一次和这一次训练的奖励策略取平均，并从头启动 3000 次迭代。任务为 RobotLab-Isaac-Roll-LeftThigh-Slope-TQBot2-v0。

## 数学定义

旧分支使用目标周期窗口（窗口长度 2π/|滚速命令|，初期使用已有样本），新分支使用最近实际净负向转过 2π 的窗口。两者都按时间平均。

```
Sr_old = exp(-((mean_roll_old - cmd_roll)/1.5)^2)
Sy_old = exp(-((mean_yaw_old - cmd_yaw)/0.25)^2)
Sr_new = valid_actual_cycle * exp(-((mean_roll_new - cmd_roll)/1.5)^2)
Sy_new = valid_actual_cycle * I(-mean_roll_new > 6 + 1e-6)
         * exp(-((mean_yaw_new - cmd_yaw)/0.125)^2)

roll_reward = 0.5 * (8*Sr_old + 8*Sr_new)
yaw_reward  = 0.5 * (5*Sr_old*Sy_old + 5*Sy_new)

reward_step = 0.02 * [roll_reward + yaw_reward
    - 1e-4*(a_world.x² + a_world.y² + 2*a_world.z²)
    - 2e-6*(alpha_world.x² + alpha_world.y² + 2*alpha_world.z²)
    - 1e-4*sum(abs(applied_joint_torque*joint_velocity))]
```

这是两套奖励输出的算术平均，不是平均误差尺度、观测或课程参数。两套正则项相同，取平均后系数保持原值。首个实际完整圈前，新分支为零，旧分支仍有基于已收集窗口的得分，因此总跟踪奖励可为旧分支的一半；不额外套一个首圈门槛。滚速恰为 −6 时新偏航分支关闭，旧偏航分支仍可贡献奖励。

## 其余训练条件

- 随机初始化 actor/critic/optimizer，resume=false，初始动作标准差 0.8；仍为高斯采样并裁剪到 ±1，未切换 tanh。
- 4096 并行环境，seed=42，3000 次 PPO 迭代，每 100 次保存。
- 命令滚速 −14～−6 rad/s；偏航从 0 开始，最终范围 ±0.5 rad/s。
- 回合验收仍为至少 6 s/4 圈，滚速误差≤20%、偏航误差≤0.06 rad/s 各自至少80%圈通过；课程批次阈值和初速、相位、坡度升级规则不改。
- 课程从辅助系数1、出生相位 −2±0.1 rad、平地开始。零初速及整圈随机相位是先前临时回放条件，没有并入训练。
- COM 每轴±3 cm、trunk质量±1 kg、其他刚体质量×0.8～1.2、Kp/Kd各×0.8～1.2；动作范围保持 4 hip + 2 左大腿。
- 通过命令配置 legacy_reward_fraction=0.5 启用；默认 hip-only 分支保持0。35维观测不变。

## 验证

28项CPU回归通过。新增验证逐步比较平均结果与两套独立历史计算器，包含非恒速、偏航变化、反向、严格6 rad/s门槛、首圈前奖励与部分环境重置；同时检查观测和课程计数不受奖励平均影响。日志见 outputs/averaged_reward_check_20260929。实际新训练效果尚待观察。

实际运行：[2026-09-29_17-16-30_avg_rewards_scratch3000](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-29_17-16-30_avg_rewards_scratch3000/)，[配置核验](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-29_17-16-30_avg_rewards_scratch3000/launch_config_verification.json)。64环境坡度smoke通过；从迭代0正常学习并保存model_0。
