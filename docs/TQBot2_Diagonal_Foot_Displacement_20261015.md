# TQBot2 对角足位移惩罚（2026-10-15）

StandardWalk 的对角协调项已改为比较足端相对默认站姿的位移。比较机身坐标系的 X/Z 分量，不约束横向 Y；两组为 FR/RL、FL/RR。保留原权重 −0.05、两对取平均、直立门控及奖励管理器的 dt 缩放。

## 定义

实际足端采用独立 foot 刚体的 link 原点（球心），不是小腿原点或刚体质心。使用 root link 的位置和姿态转换到机身坐标系：

```text
p_body[i] = inverse(R_root_link) * (p_foot_link_world[i] - p_root_link_world)
delta[i] = p_body[i] - p_standing_body[i]
C_foot = 0.5 * (
    ||(delta[FR] - delta[RL])[x,z]||²
  + ||(delta[FL] - delta[RR])[x,z]||²
)
upright = clamp(-projected_gravity_body.z, 0, 0.7) / 0.7
reward = dt * (-0.05) * k * C_foot * upright
```

默认足位由运行资产的 URDF 和 `default_joint_pos` 做正向运动学计算，不使用首次观测足位、随机重置的落脚点或策略动作零位。初始化后固定，局部 reset 不重新采样。四个参考足位约为：

| 足 | x / m | y / m | z / m |
|---|---:|---:|---:|
| FR | 0.1996978 | −0.12675 | −0.25 |
| RL | −0.1996978 | 0.12675 | −0.25 |
| FL | 0.1996978 | 0.12675 | −0.25 |
| RR | −0.1996978 | −0.12675 | −0.25 |

该项仍是软惩罚，不是步态相位约束，也不提供主动抬脚收益。

## 惩罚尺度

“与旧项幅度一致”定义为默认站姿附近同一批校准样本的平均成本一致，而非每个姿态的惩罚都相等。初始化时使用局部 NumPy 随机数发生器（seed=42），生成 8192 组独立均匀关节扰动，每个 hip/thigh/calf 为 ±0.05 rad。样本不推进物理、不消耗训练全局随机数流。

旧项按三关节角度差平方和、两对平均计算；新项按上述 X/Z 位移差计算：

```text
k = mean(C_old) / mean(C_foot)
```

本次 PhysX 运行读取实际 float32 默认关节角后得到：

- `mean(C_old) = 0.0049932217806276154`
- `mean(C_foot) = 0.00015649547378681809 m²`
- `k = 31.906493266568866`
- 加权平均成本率均为约 `−0.0002496611`，20 ms 控制步下约为 `−0.00000499322`。

系数只在初始化时确定，此后不随策略、奖励统计或 episode 自适应变化。URDF 或默认站姿改变后，新进程会重新校准。当前 URDF SHA256 为 `28c2167e55479f8701392a2010a858dce17c6a6c2f997c8172cfbd7b553de7f0`。

## 验证与生效范围

- 7 项新数值测试通过：独立 FK 对照；默认站姿零成本；对角足同向前移/抬高零成本；旧角度相等而足端反向运动时有成本；单足误差的两对平均；Y 不计入；校准及独立样本尺度、随机数独立性。
- 7 项已有站姿/参考坐标测试通过。
- 实际 PhysX GPU：4 环境、40 控制步通过。验证默认站姿、随机根位置/姿态下的坐标变换、倒置门控、动作零位独立性、局部 reset、奖励管理器加权及有限输出。Go2 仍解析为原 `joint_mirror` 函数。

为维持现有日志兼容性，TQBot2 日志键仍为 `Episode_Reward/joint_mirror`，但新配置的函数为 `FootDisplacementSymmetry`；新旧曲线的判断标准不同，不能仅按同名指标推断同一种行为。

本次更新当前源码，未切换或重启既有冻结源码训练。新逻辑在下一次使用当前源码创建 StandardWalk 环境时生效。验证确认计算与集成正确，不代表已经验证训练性能改善。

实现：[奖励项](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py)、[FK 与校准](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/foot_symmetry.py)、[TQBot2 配置](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/tqbot2/standard_walk_env_cfg.py)。

验证：[数值测试](../scripts/tests/test_tqbot2_foot_symmetry.py)、[PhysX 检查](../scripts/tests/tqbot2_foot_symmetry_smoke.py)、[实际运行结果](../outputs/TQBot2_foot_symmetry_20261015/physx_check.json)。
