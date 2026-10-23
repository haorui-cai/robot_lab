# TQBot2 当前配置：从零训练与 model_100 权重迁移

2026-10-15，按用户要求将四个大腿奖励软限位改为 **[0.33, 3.07] rad**。硬限位保持 **[0.20, 3.20] rad**，上下各保留 0.13 rad 的惩罚缓冲区。原软下限没有随硬限位更新，是因为行走任务通过 startup 事件单独指定软范围，不使用硬范围的自动 90% 缩放。

## 两组共同条件

- 任务：`RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0`；每组 4096 环境、seed 42、12000 次新更新、每 100 轮保存。
- 同一份冻结源码、新 `TQBot2_walk.urdf`、独立 foot 刚体，保留选择性腿间物理碰撞。
- 隐式 PD 名义 20/0.5，reset 增益随机化 0.5–2 倍；力矩上限 90%，髋/大腿 21.33、小腿 31.995 N·m。
- foot/calf/thigh 不计非期望接触惩罚，trunk/hip 仍计；对角协调使用 `FootDisplacementSymmetry`，权重 −0.05。
- 球面接触点滑移 −0.1；机身相对足高目标 −0.165 m、权重 −5，仅离地足计算。步态 0.5、腾空阈值 0.5 s 保留。
- 策略参考 hip=0、thigh=0.8、calf=−1.6445578923423 rad；站立参考与策略参考分别定义。
- PPO 初始学习率 0.001、自适应调节，熵 0.01；每环境每次采样 24 步。物理 200 Hz、策略 50 Hz。
- 六类粗糙地形及原课程、随机化、命令范围保持当前设置；不加入高摩擦专用平地。

## 初始化差别

| 组别 | 初始化 | 后续训练 |
|---|---|---|
| scratch | 随机 actor/critic，新 Adam | 当前新配置，从 0 编号，共 12000 次更新 |
| warmstart100 | 继承指定 model_100 的 actor/critic（含探索标准差），新 Adam | 同一当前新配置，从 0 编号，共 12000 次更新 |

源检查点：[model_100.pt](/home/eureka/robot_lab/logs/rsl_rl/tqbot2_rough/2026-10-12_22-24-51_TQBot2_foot_ground_clearance3cm_from_gentle500/model_100.pt)。原文件不改写，另存副本并记录 SHA-256。

继承组采用 `--warm_start`，不恢复旧环境、旧奖励、旧学习率、旧优化器或旧迭代编号。actor 输入 45 维、critic 输入 235 维、输出 12 个动作，与新任务匹配；实际加载采用严格权重形状检查。保留原网络数值，不做坐标变换：旧小腿观测/动作参考为 −1.5 rad，新参考为 −1.6445578923423 rad。因此它是旧权重在新任务上的迁移训练，不承诺保持原策略的逐步物理目标或轨迹。新模型、奖励、参考坐标、目标裁剪和接触定义都由当前配置决定。

两组环境/PPO 设置相同，仅初始化不同；单一 seed 对照不能量化跨种子的稳定收益。当前检查只验证实现和启动，不表示行走效果已改善。

## 验证和输出

7 项站姿/参考坐标测试、7 项足端位移数值测试通过；旧 model_100 的 actor 在 16 个生成地形环境完成 300 步集成检查，核对了实际软硬限位、PhysX PD、接触惩罚选择和局部 reset，观测/奖励均有限。

- [实验清单](/home/eureka/robot_lab/outputs/TQBot2_soft033_dual_20261015/experiments.json)
- [从零组启动记录](/home/eureka/robot_lab/outputs/TQBot2_soft033_dual_20261015/scratch/launch_metadata.json)
- [继承组启动记录](/home/eureka/robot_lab/outputs/TQBot2_soft033_dual_20261015/warmstart100/launch_metadata.json)
- [集成检查](/home/eureka/robot_lab/outputs/TQBot2_soft033_dual_20261015/smoke.json)
- [共同源码哈希清单](/home/eureka/robot_lab/outputs/TQBot2_soft033_dual_20261015/source_manifest.json)

两组均已启动并保存 model_0。15:29 核验时，从零组记录到第 20 轮、继承组到第 2 轮；移除日志目录及运行名称后，实际保存的环境和 PPO 配置完全一致，冻结源哈希有效，继承记录确认加载指定 model_100 的副本。见 [运行配置核验](/home/eureka/robot_lab/outputs/TQBot2_soft033_dual_20261015/config_verification.json) 与 [双组对比看板](http://127.0.0.1:6006/?run=8352710db433282e,3c7b30283baf11a6)。

## Go2 暂停

按用户指令暂停 Go2 并释放 GPU。暂停前最后记录迭代 10232，最新完整检查点为 model_10200；未保存的后续更新不能从该文件恢复。保留日志、权重与源码快照，不自动重启 Go2。详见 [暂停记录](/home/eureka/robot_lab/outputs/Go2_rough_official_resume_to15000_20261014_223152/PAUSED.json)。
