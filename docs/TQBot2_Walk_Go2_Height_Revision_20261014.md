# TQBot2 StandardWalk 使用 Go2 原高度项（2026-10-14）

同日后续已切换 StandardWalk 为显式 DC、取消额外大腿目标裁剪、改用 Go2 防滑项并移除高摩擦平地，见 [执行器与防滑对齐](TQBot2_Walk_DC_Go2_Alignment_20261014.md)。本文保留前一轮修改记录。

按用户批注调整仓库 `TQBot2StandardWalkRoughEnvCfg`，后续从该源码启动时生效；历史训练快照和保存参数保留。

- 恢复 Go2 使用的 `feet_height_body`：同一奖励函数，权重 -5.0、机身坐标高度目标 -0.2 m、`tanh_mult=2.0`、命令 `base_velocity`。身体选择适配 TQBot2 的 `.*_calf`，不使用 Go2 的 `.*_foot`。
- 关闭此前作为替代的 `feet_swing_clearance`，避免同时施加两种高度目标。真实足端地面滑移项继续使用。
- 删除 Go2 原配置没有的两项额外能耗奖励：`walking_energy_efficiency`、`walking_joule_proxy`。Go2 原有的 `joint_torques_l2`（-2.5e-5）和 `joint_power`（-2e-5）保留。
- 保留上一轮已调整的步态权重 0.5、三项恢复成本关闭、较低阶梯/方块范围，以及其他已有设置。
- 更新既有集成检查脚本，检查原高度项开启、替代间隙项和额外能耗项关闭。

Go2 的 `randomize_reset_base` 在每次复位时随机化 roll/pitch/yaw，各范围 (-3.14, 3.14) rad。范围包含倒置与侧倒姿态；这是欧拉角分别均匀采样，不是专门固定的倒置组，也不是 SO(3) 均匀采样。Go2 同时关闭非法接触终止，保留直立奖励，因此训练可从倒地状态继续学习恢复。TQBot2 StandardWalk 使用相同姿态范围。

两种执行器都实现位置 PD。Go2 的显式模型先计算力矩，再按速度相关电机边界裁剪并将力矩交给 PhysX；TQBot2 的隐式模型把目标位置及 PD 增益交给 PhysX 的关节驱动求解。差异在仿真求解方式，而非 TQBot2 不使用 PD。

验证输出位于 `outputs/TQBot2_walk_go2_height_revision_20261014/`：实际构造 Go2 和 TQBot2 配置，对比高度奖励函数、权重与参数、力矩/功率成本、复位姿态范围，同时核对此前保留的地形及动作设置。源码和集成检查脚本进行语法检查。本次不启动训练或物理步进，不声称策略性能已验收。
