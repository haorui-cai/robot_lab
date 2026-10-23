# TQBot2 策略零位与站立参考分离（2026-10-14）

最新选择：StandardWalk 默认改用旧策略零位（大腿 0.80 rad），隐式 PD、粗糙地形、12000 轮从零训练。新零位 1.35 rad 仍保存在 `WALK_POLICY_ZERO_JOINT_POS`；活动旧零位另存于 `WALK_LEGACY_POLICY_ZERO_JOINT_POS`，动作和 actor/critic 位置观测同步使用它。下文保留新零位设计及此前验证记录。

当前 StandardWalk 将 action=0 的关节目标角与站立参考分别配置。actor 和 critic 的关节位置观测均使用策略零位，各自的关节顺序保持不变。

同日后续已将 StandardWalk 执行器切回隐式 PD，并通过本文件所述零位检查；限位成本及新零位影响见 [隐式 PD 与策略零位分析](TQBot2_Implicit_PD_Policy_Zero_20261014.md)。

| 用途 | hip / rad | thigh / rad | calf / rad |
|---|---:|---:|---:|
| 策略零位：动作偏置、关节位置观测基准 | 0 | 1.35 | -1.6445578923423 |
| 站立参考：资产默认关节角、复位、姿态惩罚 | 0 | 0.80 | -1.6445578923423 |

`assets/tqbot_walk_stance.py` 中的 `WALK_STANDING_JOINT_POS` 与 `WALK_POLICY_ZERO_JOINT_POS` 为独立字典。大腿策略零位按后续要求取 `(1.90 + 0.80) / 2 = 1.35 rad`，作为独立常量保存；策略 hip/calf 零位显式固定为本次修改前的数值，所有策略零位均不随站立参考变化。旧名称 `WALK_JOINT_POS` 保留为站立参考的兼容别名。

`TQBOT2_WALK_CFG.init_state.joint_pos` / `default_joint_pos` 继续保存站立参考。StandardWalk 的标准 `JointPositionAction` 设置 `use_default_offset=False`，使用独立的策略偏置，不添加新的动作裁剪或执行器。

```
q_target = q_policy_zero + scale * action
joint_position_observation = q_actual - q_policy_zero
```

动作尺度仍为 hip 0.125、大腿/小腿 0.25。站立时大腿动作 `(0.80 - 1.35) / 0.25 = -2.2`；站立时大腿关节位置观测为 -0.55 rad。当前 PPO 包装器的 `clip_actions=None`，不会将此动作截断到 [-1,1]。环境已有的目标角裁剪 [-100,100] rad 保留。

## 奖励参考

| 项目 | 权重 | 参考 |
|---|---:|---|
| `joint_pos_penalty` | -1 | 实际关节角相对站立参考的 L2 范数；静止倍率仍为 5 |
| `stand_still` | -2 | 小命令时，实际关节角相对站立参考的 L1 距离 |
| `feet_height_body` | -5 | 站立足端 Z=-0.25 m，加 0.085 m，目标仍为 trunk 相对 -0.165 m |
| `joint_pos_limits` | -5 | 实际关节角相对软限位的越界量，与上述参考独立 |

奖励函数及其门控、权重保持。前两项原本读取 `default_joint_pos`，因此将它保留为站立参考即可；无需另写奖励函数。足高仍使用现有站立几何常量，不由策略零位计算。旧行走、滚动与恢复任务不使用新的 StandardWalk 观测配置。

## 限位与图

以下数值从当前 `TQBot2_walk.urdf` 读取；四腿同类关节数值相同。软范围依据资产的 0.9 系数计算，其中心与硬范围相同。

| 关节 | 硬范围 / rad | 软范围 / rad | 策略零位 | 从硬下限算起的位置 |
|---|---|---|---:|---:|
| hip | [-1.588, 1.588] | [-1.4292, 1.4292] | 0 | 50% |
| calf | [-2.975, -0.916] | [-2.87205, -1.01895] | -1.6445578923423 | 64.62% |
| thigh | [0.60, 3.20] | [0.73, 3.07] | 1.35 | 28.85% |

小腿范围中点是 -1.9455 rad；当前策略零位偏向数值更大的上限侧，距硬下限 1.3304421 rad，距硬上限 0.7285579 rad。

绘图脚本：`scripts/tools/plot_tqbot2_policy_zero.py`。图和带 URDF 哈希的原始数值位于 `outputs/TQBot2_policy_zero_decoupling_20261014/`。各行按自身硬范围归一化显示位置，不代表同一弧度比例或足端轨迹。

## 验证与生效范围

`scripts/tests/test_tqbot2_walk_stance.py` 的 6 项检查通过，覆盖真实 URDF 正运动学、站立/策略零位限位、四腿大腿策略零位为前两次零位的中点及共享资产参考。

`scripts/tests/tqbot2_policy_zero_smoke.py` 在 CPU PhysX 的 4 个环境中通过，验证真实 ActionManager 和 ObservationManager 的关节顺序、零动作目标、站立动作 -2.2、actor/critic 观测、两项奖励的站立最小值、独立修改动作偏置不改变奖励参考、命令门控、足高目标、部分复位及 40 步有限观测/奖励。

测试中特意将实际关节角设为策略零位：四个大腿各偏离站立角 0.55 rad。零命令下 `stand_still` 原始项为 2.2，`joint_pos_penalty` 原始项为 5.5；运动命令下分别为 0 和 1.1，符合原有公式。说明策略零位并未被误用为站立奖励目标。

本次未启动训练或改写已有训练快照。新的动作/观测坐标用于后续从当前源码启动的 StandardWalk 环境；旧权重直接配合新环境会改变行为，不能视为等价续训。本次检查验证参考语义和环境集成，不证明新零位下的训练收敛或行走性能。
