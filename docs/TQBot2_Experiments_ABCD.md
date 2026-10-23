# TQBot2：A/B/C/D 实验身份与对照

2026-09-29，按用户指定命名。后续讨论中的 A、B、C、D 固定对应以下实验，不能只凭同名 model_11999.pt 判断来源。

| 实验 | 目录（logs/rsl_rl/tqbot2_roll_left_thigh_slope/ 下） | 初始化 | 跟踪奖励 | 迭代计划 |
|---|---|---|---|---|
| A | 2026-09-24_21-03-39_base_accel_world_z2_from_slope5600 | 历史坡度模型续训 | 目标周期窗口；滚速权重8；偏航权重5、尺度0.25、乘滚速得分 | 已完成，最终 model_11999.pt |
| B | 2026-09-29_00-40-27_tight_yaw_narrow_dr_actual_cycle_scratch | 从头训练 | 实际周期窗口；滚速权重8；偏航权重5、尺度0.125、负向滚速>6门槛 | 已完成12000次，最终 model_11999.pt |
| C | 2026-09-29_17-16-30_avg_rewards_scratch3000 | 从头训练 | A与B的奖励输出各占50%，保留各自窗口 | 3000次 |
| D | 2026-09-29_17-18-32_D_Breward_yaw10_from_A11999 | 加载A的model_11999.pt，恢复模型及优化器 | B的奖励，偏航权重从5提高至10；不使用平均奖励 | 追加3000次 |

A 最终模型 SHA256：`65283a7a96d5a365e1abd34e394213f0b734a09f42d2ce670fca801b5adefa81`。D 启动前已核对该哈希。

## D 的有效配置

```
legacy_reward_fraction = 0.0
actual_cycle_rewards = true
roll_reward = 8 * exp(-((mean_roll_actual - cmd_roll)/1.5)^2)
yaw_reward = 10 * I(-mean_roll_actual > 6 + 1e-6)
             * exp(-((mean_yaw_actual - cmd_yaw)/0.125)^2)
```

首个实际净负向完整圈前，两项跟踪奖励为0。保持原有三项加速度/机械功率正则，乘控制步长0.02后发放。课程单圈偏航门槛仍为0.06 rad/s，不因权重加倍而放宽。

D 的其余环境使用 B/C 的最新窄物理随机化：COM每轴±3 cm、trunk质量±1 kg、其他刚体质量×0.8～1.2、Kp/Kd各×0.8～1.2。4096环境、seed42。初始策略及探索标准差来自 A，不重新初始化；课程在新环境中从零级重建（平地、零偏航、相位−2±0.1、辅助系数1），A结束时也处于零级。

C 的默认配置不改动。D 使用启动命令覆盖混合比例和偏航权重，因此两个进程可以独立运行。为给 C/D 腾出显存，关闭此前零初速随机相位的 GUI 回放。两个训练共享同一 GPU，各自保存 YAML、日志、检查点、源码快照和启动元数据。

C 和 D 的初始化、奖励不同，不能将二者差异单独归因于偏航权重。D 用于回答“从 A 已有能力出发，在 B 的奖励结构下增加偏航权重，能否改善偏航”；需要同时检查滚速、偏航及联合成功率。

- [C 日志](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/console_20260929_avg_rewards_scratch3000.log)
- [D 日志](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/console_20260929_D_Breward_yaw10_from_A.log)
- [D 启动命令](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/launch_20260929_D_Breward_yaw10_from_A.sh)

D 启动验证：保存的 env.yaml/agent.yaml 逐项核验通过；确认加载A路径，从11999次继续更新并保存model_12000。C同时运行且其保存配置仍为混合比例0.5、偏航权重5。初始并行GPU显存约12.4/16.4 GB，迭代时间约6.8 s，效果需后续评估。
