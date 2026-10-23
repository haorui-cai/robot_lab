# TQBot2 StandardWalk 站姿与足高参考帧（2026-10-14）

当前 StandardWalk 的名义关节角为 hip=0、thigh=0.80、calf=-1.6445578923423 rad。按用户后续要求，默认 foot 原点相对 trunk 的 Z 固定为 -0.25 m，足高目标为默认位置 +0.085 m，即 -0.165 m。前足相对 hip 的 X 为 -38.40 mm，后足为 +38.40 mm；比最初 0/0.8/-1.5 站姿的 54.23 mm 偏移更靠近 hip 下方。左右腿的 XZ 投影相同。

**后续策略零位解耦：**上述角度现在明确作为站立、复位及奖励参考。策略动作零位的大腿设为 1.35 rad（此前的 1.90 与原始 0.80 的中点），hip=0、小腿=-1.6445578923423 rad；actor/critic 关节位置观测也相对策略零位。站立角、足高目标和前三项奖励的参考均保持上述数值。实现与验证见 [策略零位与站立参考分离](TQBot2_Policy_Zero_Decoupling_20261014.md)。

本版替代此前更蹲的 0.85/-2.10 站姿（默认 foot Z=-0.188820732 m）。由于小腿比大腿短，在 Z=-0.25 m 的约束下，维持较小的水平偏移需要进一步减小大腿角并靠近限位。本版保留最初大腿0.80及其0.07 rad软下限余量，通过求解小腿角满足指定高度。全部关节位于软范围内。原出生根高度0.35 m保留，它是放置高度，不是落地后支撑高度。

## 配置范围

纯几何常量位于 `source/robot_lab/robot_lab/assets/tqbot_walk_stance.py`，包含 WALK_JOINT_POS、默认 foot Z，以及 0.085 m 的提升量。`tqbot.py` 在公共资产的独立副本上构造 TQBOT2_WALK_CFG；StandardWalk 使用该副本。公共 TQBOT2_CFG 的 0/0.8/-1.5 继续供旧行走、滚动等任务使用，恢复任务保留自己的姿态覆盖。

只修改当前 StandardWalk 默认关节角和足高奖励目标；奖励函数、-5 权重、tanh_mult=2、接触选择、PD、力矩、动作尺度、随机化、硬限位及 PPO 未变。此次没有重新启动训练或修改实机参数。17:00 启动的平地实验使用独立源码快照，其保存配置保留旧关节角和旧 -0.2 足高目标。

## 足高目标

足高项使用 foot 连杆原点，默认原点与 TQBot2 足球球心重合，不是球面接触点。

```
z_default_trunk = -0.25 m
thigh = 0.80 rad
calf = -thigh - acos((0.25 - 0.213*cos(thigh))/0.153)
     = -1.6445578923423 rad
z_target_trunk = z_default_trunk + 0.085 = -0.165 m
```

这表示在同一 trunk 坐标系中高于默认足端位置 8.5 cm；不直接规定世界地面间隙。原函数仍用相对机身的脚部水平速度加权，没有额外添加摆动脚接触门控。

## trunk 与 base 的核查

`feet_height_body` 减去 `asset.data.root_pos_w` 并使用 `root_quat_w` 的逆旋转。当前 IsaacLab 中，这两个字段分别是 root_link_pos_w / root_link_quat_w 的别名，body_pos_w 是 body_link_pos_w 的别名。因此高度是相对 articulation 根连杆原点，不是整机质心或根刚体 COM。配置中的 base_link_name 本身不会改变这个函数的参考点。

TQBot2 的训练 URDF 删除了无质量虚拟 base 和 floating_base，实际根为 trunk。Go2 本地 URDF 的实际根为 base，因此同一函数在 Go2 中相对 base。

TQBot2 原始 ROS URDF 的 floating_base 为 base→trunk：(0, -0.0000611544755962, +0.0201840868229783) m，旋转为零。等价坐标如下：

| 量 | trunk 坐标 | 原始虚拟 base 坐标 |
|---|---:|---:|
| 新默认 foot Z | -0.250000000 m | -0.229815913 m |
| 同一足高目标 Z | -0.165000000 m | -0.144815913 m |
| 相对默认位置提升 | +0.085 m | +0.085 m |

当前奖励继续使用 trunk，参数已按 trunk 计算，没有额外套用 base 偏移。

## 验证

4 项 CPU 检查通过：对实际 URDF 逐关节正运动学验证四脚水平偏移、8.5 cm 高度差、软限位、原始 base 与导入 trunk 的坐标关系及公共资产姿态保持。

16 环境、300 步 PhysX 集成检查通过。在首次物理积分前实际读取四个 foot 的 root-relative Z，约 -0.25 m，与 URDF FK 一致；实际奖励目标 -0.165 m。包含独立足端、自碰撞配对、显式执行器、有限奖励和部分环境复位检查。该检查只读取旧 model_2999 权重且不更新，不能作为新站姿的行走、恢复或数值稳定性验收。

图、正运动学数据及原始物理检查日志位于 `outputs/TQBot2_walk_stance_z025_20261014/`。旧图和实验配置保持历史记录。
