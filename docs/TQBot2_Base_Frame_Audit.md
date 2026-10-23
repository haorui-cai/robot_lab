# TQBot2 base 与 trunk 参考帧核对

2026-09-24 早期核对结论（后续修订见下）：同步的 xacro 是最新的；Isaac 适配移除了源模型中的虚拟 base，因此仿真 root 是 trunk，二者不能混用名称。此前诊断把 root/trunk 曲线标为 base，这是诊断参考点命名错误。

## 后续修订：新 base 与加速度正则

本地源模型提交 `0211114e90615e96e793fc5deebfa2a9eb286166` 将 base→trunk 改为 (0, -0.0000611544755962, +0.0201840868229783) m。已仅同步 `floating_base` 固定关节至训练仓库的原始 URDF/xacro，并更新同步清单；物理 Isaac URDF、派生模型和出生高度表保持不变。下文 24.7 mm 和诊断数值均描述修订前的参考点，不能作为新 base 的曲线。

新训练配置已添加新 base 的世界系线加速度/角加速度正则，Z 系数为 X/Y 的 2 倍。现有训练进程不热更新，仍为旧奖励。诊断脚本从原始 URDF 读取参考点，因此后续重新采样会自动使用新 base；已有图和 CSV 保持原历史数据。

## 版本和加载链（早期快照）

- `git ls-remote origin refs/heads/TQBot2` 返回 `ab9b99cce226930865a9d3df25c00a6d3df2c422`，与 `sync_manifest.json` 的来源提交一致。
- unitree_guide 本地 TQBot2 HEAD 为 `5de838769c002f2f8317302e8c106e5f2aa14195`，比远端多一个控制器提交；该提交没有修改 `TQBot2_description`。
- 训练目录中全部 12 份 xacro 与本地 HEAD 对应文件逐字节一致。
- 原始 `TQBot2.urdf` 根为 `base`；`TQBot2_isaac.urdf` 和当前训练加载的 `TQBot2_left_thigh_roll.urdf` 根为 `trunk`。
- 运行实验的 `params/env.yaml` 指向 `TQBot2_left_thigh_roll.urdf`，文件 SHA256 为 `c225bf9dd8a167bf4e7363c85d4af1bbff55141b3c493819af440cf8a4827833`，与派生模型清单一致。
- 5 项模型同步/派生测试通过，物理几何、惯性及关节变换的保留关系与原适配设计相符。

## base 的定义

`xacro/robot.xacro` 的 `floating_base` 固定关节以 `base` 为父、`trunk` 为子，平移为 `(0, 0, +0.0247) m`、旋转为零。注释明确说明 base 是滚动外壳中心，比旧 trunk 原点低 24.7 mm。

同步脚本 `scripts/tools/sync_tqbot2_description.py` 第 54～58 行删除了无质量虚拟 `base` 及固定关节 `floating_base`，保留所有 trunk 相对的实体几何及惯性。这个适配没有移动实体之间的相对位置，但改变了仿真根参考帧。

正确的参考点换算（世界坐标）：

```text
r_trunk_to_base = [0, 0, -0.0247]
r_world = R_world_trunk * r_trunk_to_base
p_base_world = p_trunk_world + r_world
v_base_world = v_trunk_world + omega_world × r_world
a_base_world = a_trunk_world + alpha_world × r_world
               + omega_world × (omega_world × r_world)
```

因此不能把 base 的世界 Z 简单写成 trunk 的世界 Z 减 0.0247，也不能把两点的线加速度视为相同。角速度在同一刚体上的这两个参考点相同。

## 诊断更正

`diagnose_roll_vertical_motion.py` 已从原始 ROS URDF 读取固定变换，记录真正 `xacro_base_*` 的世界位置、速度及速度差分加速度，并保留 `base_link_*` 作为历史列名（其含义是 Isaac root/trunk）。绘图脚本要求新数据中包含 `xacro_base_*`，拒绝再把旧 CSV 当作正确 base 曲线绘制。

修正后采样目录：`outputs/roll_vertical_diagnostics/model_4700_xacro_base/`。为便于对照，仍使用同一个 model_4700、seed 42、三个环境和 -6/-10/-14 rad/s 命令。旧图与数据保留，旧 README 已加更正。

重新采样完成 2000 个物理步、6003 行数据。验证 base 与 trunk 的世界 Z 差不超过 24.7 mm，且整机质心和 trunk 轨迹与旧采样一致。剔除回合前 0.5 秒后的高度峰峰值：

| 滚速命令 | 旧图 trunk | 真正 xacro base |
| --- | --- | --- |
| -6 rad/s | 43.6 mm | 15.9 mm |
| -10 rad/s | 81.1 mm | 68.8 mm |
| -14 rad/s | 73.3 mm | 64.1 mm |

不同命令对应独立随机化环境，不作为严格的速度因果比较。-10 rad/s 样本真正 base 的加速度绝对峰值为 221.5 m/s²（5 ms），48.0 m/s²（20 ms）；近似离地比例仍为 26.8%。曲线文件 `base_z_diagnostic.png` 中蓝线现为 xacro base，灰虚线为 trunk，橙线为整机质心。

原先“当前没有竖直线加速度惩罚”仍成立；整机质心和接触力用于判断腾空的结论不依赖把 trunk 误标成 base。当前训练仍沿用既有 trunk 根参考帧，本次没有更换训练物理模型。
