# TQBot2 当前滚动训练配置总览

**2026-10-14 后续训练关节摩擦更新：**按用户要求，TQBot2 导入时将旧 PhysX 关节摩擦系数清零，仅使用新接口；全部活动关节静摩擦力矩 **0.4 N·m**、动摩擦力矩 **0.4 N·m**、黏性摩擦系数 **0**。共享配置覆盖行走、滚动和恢复，StandardWalk 显式 DC 转换也复制全部三项。6 类模型各 4 个 GPU 环境副本的实际 PhysX 数值、步进/复位及新摩擦全零对照已验证，见 [修改与验证](TQBot2_New_Joint_Friction_20261014.md)。下文历史实验参数不改写。

**2026-10-14 后续训练默认增益更新：**按用户要求，TQBot2 共享机器人配置的全部关节名义 Kp/Kd 改为 **14/0.6**，行走、滚动和恢复任务统一继承。原有增益随机化倍率保留；下文历史实验的 20/0.5 及其随机范围仍表示当时保存的配置，已有模型和运行不随源码更新。此次 5 ms 固定目标稳定性对照见 [14/0.6 诊断](TQBot2_Kp14_Kd06_Stability_20261014.md)。

核对更新：2026-09-29。对象为 `RobotLab-Isaac-Roll-LeftThigh-Slope-TQBot2-v0`，即四 hip＋两个左大腿、新 base 加速度正则、上下坡课程。当前运行见下方 09-29 记录；第 1～9 节主体保留 model_11999 历史实验说明，当前变更以页首更新和新实验保存的 YAML 为准。

**2026-09-29 新增并行实验D：**按用户定义，旧训练为A、新的12000次从头训练为B、奖励平均后的3000次训练为C。D从A的最终model_11999恢复模型和优化器，使用B奖励并将偏航权重5→10，追加3000次；通过命令行将 `legacy_reward_fraction=0.0`，不改变C的默认0.5。其余环境沿用最新配置，详见 [A/B/C/D身份与配置](TQBot2_Experiments_ABCD.md)。

**2026-09-29 当前运行：两套奖励等权平均、从头训练3000次。**用户要求将旧训练与刚完成训练的奖励各乘0.5再相加，当前左大腿及坡度配置启用 `legacy_reward_fraction=0.5`。旧分支保留目标周期窗口、偏航尺度0.25及滚速得分相乘；新分支保留实际周期窗口、偏航尺度0.125及>6门槛。两套相同的三项正则平均后不变。课程判据及物理随机化沿用最新值，临时回放的零初速/整圈随机相位不用于训练。[完整公式与条件](TQBot2_Averaged_Reward_Training.md)。已启动 [2026-09-29_17-16-30_avg_rewards_scratch3000](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-29_17-16-30_avg_rewards_scratch3000/)，resume=false，初始动作标准差0.8，4096环境；保存配置逐项核验通过，已从迭代0正常学习并保存model_0。28项CPU回归及64环境坡度smoke通过。

**2026-09-29 已完成的12000次从头训练（平均奖励之前）：**按用户明确要求，已停止先前误启动的 `00-33-43_*_from11999` 续训任务，另开 [2026-09-29_00-40-27_tight_yaw_narrow_dr_actual_cycle_scratch](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-29_00-40-27_tight_yaw_narrow_dr_actual_cycle_scratch/)。Actor、Critic 和优化器均全新初始化，`resume=false`，不加载 checkpoint；从迭代 0 开始，初始探索标准差 0.8。4096 环境、seed 42、总计 12000 次迭代，每 100 次保存。已观察正常参数更新并保存 model_0/model_100，初始日志不代表最终收敛效果。

最新实际整圈奖励、偏航门控 >6、尺度 0.125/成功阈值 0.06、COM 每轴 ±3 cm、质量和增益收窄均已生效；保存的 [env.yaml](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-29_00-40-27_tight_yaw_narrow_dr_actual_cycle_scratch/params/env.yaml)、[agent.yaml](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-29_00-40-27_tight_yaw_narrow_dr_actual_cycle_scratch/params/agent.yaml) 和 [逐项核验](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-29_00-40-27_tight_yaw_narrow_dr_actual_cycle_scratch/launch_config_verification.json) 已检查。课程从零级开始：坡度 0、偏航命令 0、相位半宽 0.1、初速辅助 1；策略仍为高斯采样加动作裁剪。见 [启动信息](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-29_00-40-27_tight_yaw_narrow_dr_actual_cycle_scratch/launch_metadata.json) 与 [训练日志](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/console_20260929_current_scratch.log)。相关源码、资产与文档已归档到该实验的 training_source_snapshot，含文件哈希；这不是完整软件环境镜像。误启动续训的日志保留作历史，已在其目录标记 STOPPED.json。

**2026-09-27 后续配置变更：**用户要求将 trunk COM 随机偏移改为 x/y/z 各 ±0.03 m，已覆盖在左大腿课程配置中，坡度子类继承。此修改供下次启动/续训使用，尚未启动新训练；下文 `model_11999` 的历史实验仍是 ±0.10 m，历史 YAML 与模型不改写。

**2026-09-28 后续物理随机化变更（当前代码默认）：**左大腿及坡度任务的 Kp/Kd 各乘均匀 0.8～1.2，即 Kp 16～24、Kd 0.4～0.6；trunk 质量加 −1～+1 kg；其他合并刚体质量乘 0.8～1.2。质量仍在 startup 抽样并重算惯量，增益仍在 reset 抽样，COM 各轴仍为 ±0.03 m。供下次启动/续训使用，尚未启动新训练；下文随机化表保留 model_11999 的历史实际设置。已在 IsaacLab 中展开两个任务的配置，确认范围、目标刚体/关节和采样时机一致；[展开记录](../outputs/physical_randomization_check_20260928/resolved_config.json)。

**2026-09-28 后续奖励变更（当前代码默认）：**左大腿及坡度任务启用 `actual_cycle_rewards=true`，两项跟踪奖励取最近实际净负向滚过 2π 的时间窗口；偏航奖励改为 `5 * I(-mean_roll > 6) * exp(-((mean_yaw-cmd_yaw)/0.125)^2)`，不再乘滚速跟踪得分。这里 6 是命令范围 −14～−6 的最小目标方向速度幅值，用户已确认严格大于；数值比较保留 1e-6 容差。首圈完成前两项跟踪奖励为零。偏航尺度进一步按用户要求由 0.25 收紧到 0.125 rad/s：误差 0.05 时保留 85.214% 得分。单圈偏航成功阈值由 ≤0.15 收紧到 ≤0.06 rad/s（浮点边界容差 1e-7 rad/s），同时用于滚动和坡度课程。其他权重、三项惩罚、回合/批次通过比例和初速课程不变，未重启训练。历史 hip-only 默认尺度/阈值仍为 0.25/0.15。

实际一周窗口通过积分机身负 x 角速度查找最近净转角 2π 的历史边界，在控制步内线性插值，再分别计算滚角/时长和世界偏航角积分/时长。反向转动抵消净进度，停止时间计入窗口，不保持上一个高速圈的旧奖励。历史覆盖完整 10 s 回合；部分环境重置只清除对应环境。

为保持既有模型和 Guide 接口的观测含义，35 维观测中的 `rolling_cycle` 仍保留原目标周期均值，只有奖励改为实际周期均值。历史 hip-only 默认关闭此新选项，保持原行为。24 项 CPU 回归与 64 环境 Isaac 坡度 smoke 已通过；[验证记录](../outputs/actual_cycle_reward_check_20260928/)涵盖变速/非整数边界、反转、停转、首圈零奖励、部分重置和 >6 偏航门槛。随后本次偏航尺度/成功阈值修改通过 26 项 CPU 回归及 64 环境 Isaac 坡度 smoke，包含 ±0.05 得分、±0.06 包含边界、±0.0601 不通过及运行时配置传递；[本次验证记录](../outputs/yaw_threshold_check_20260928/)。尚未验证新奖励的训练效果。

## 1. 实验身份与同步边界

- 实验目录：[2026-09-24_21-03-39_base_accel_world_z2_from_slope5600](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-24_21-03-39_base_accel_world_z2_from_slope5600/)。已完成至迭代 11999，不再是“刚启动”的状态。
- Guide 的 RL_Roll 当前测试模型：[model_11999.pt](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-24_21-03-39_base_accel_world_z2_from_slope5600/model_11999.pt)，SHA256 `65283a7a96d5a365e1abd34e394213f0b734a09f42d2ce670fca801b5adefa81`。
- **当次实际配置**：[env.yaml](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-24_21-03-39_base_accel_world_z2_from_slope5600/params/env.yaml)、[agent.yaml](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-24_21-03-39_base_accel_world_z2_from_slope5600/params/agent.yaml)。训练入口会自动保存展开后的配置，包含继承结果和命令行覆盖。它们比只看某一个父类配置更可靠。
- **运行结果**：[控制台日志](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/console_new_base_from_slope5600.log)。课程等级和训练指标以日志为准，不能从配置的最大值推断。
- **来源**：[resume_source.json](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-24_21-03-39_base_accel_world_z2_from_slope5600/resume_source.json)、[当次源码快照](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/2026-09-24_21-03-39_base_accel_world_z2_from_slope5600/training_source_snapshot/)。快照只有 5 个指定文件，不是整个训练依赖树；首次核对时 5 个文件哈希全部一致，随后左大腿配置增加了上述 ±3 cm COM 覆盖，其余 4 个仍一致。其余源码解释按核对时工作树，不能追认成当时完整归档。

本文为人工核对的说明，**不是自动随代码改写的文档**。新增 [同步清单](TQBot2_Current_Training_Config.sources.json) 与 [只读检查工具](../scripts/tools/check_tqbot2_training_doc.py)，记录相关代码、模型描述、碰撞网格、配置快照和本文的 SHA256。从仓库根执行：

```bash
python3 scripts/tools/check_tqbot2_training_doc.py
```

文件变化会列出差异并返回非零。检查通过只证明这些文件未偏离本次核对基线，不证明文档语义自动正确、依赖包完全相同或训练可逐位复现。修改后应重新核对本文再更新清单，不能只刷新哈希掩盖过期说明。旧三份专项文档保留历史，当前参数优先查本页和本实验快照。

## 2. 环境、时间尺度与物理模型

| 项目 | 本次实际设置 |
|---|---|
| 算法/运行 | RSL-RL PPO；4096 并行环境；seed 42；cuda:0；headless |
| 仿真 | Isaac Sim / IsaacLab / PhysX，GPU 仿真 |
| 物理步长 | 0.005 s，200 Hz |
| 策略控制 | decimation=4，0.02 s，50 Hz；动作在四个物理步保持 |
| 回合 | 最长 10 s / 500 控制步；提前终止见第 7 节 |
| 重力 | 世界 `(0,0,-9.81)` m/s² |
| 机器人 | `TQBot2_left_thigh_roll.urdf`，6 自由度；固定关节合并，保留合并质量/惯量/碰撞 |
| 碰撞 | convex_hull；关闭自碰撞；环境之间过滤机器人碰撞；启用接触传感器 |
| 求解 | PhysX solver_type=1；机器人位置迭代 4、速度迭代 0；CCD 关闭 |
| 阻尼/接触 | 刚体线/角阻尼 0；最大去穿透速度 1 m/s；接触传感器每 5 ms 更新，历史 5 帧 |
| 执行器 | ImplicitActuator 位置 PD；Kp=20、Kd=0.5；力矩限幅 23.7 N·m、速度限幅 30.1 rad/s、关节摩擦参数 0.2 |
| 软限位 | URDF 位置范围的 0.9 倍；当前没有软限位奖励，不能把软限位当作目标裁剪 |

训练日志记录硬件为 RTX 4070 Ti SUPER，驱动 580.178.04、Ubuntu 22.04.5。2026-09-27 查询当前 `isaaclab` conda 环境：Python 3.11.15、Isaac Sim 5.1.0.0、IsaacLab 0.54.3、isaaclab-rl 0.5.1、rsl-rl-lib 5.0.1、PyTorch 2.7.0+cu128、NumPy 1.26.0。**这是核对时环境，不是当次训练保存的完整依赖锁文件**。

资产：[派生 URDF](../source/robot_lab/data/Robots/TQBot/TQBot2_description/urdf/TQBot2_left_thigh_roll.urdf)、[派生清单](../source/robot_lab/data/Robots/TQBot/TQBot2_description/left_thigh_roll_manifest.json)、[出生高度表](../source/robot_lab/data/Robots/TQBot/TQBot2_description/left_thigh_roll_spawn_heights.json)。派生 URDF SHA256 `c225bf9dd8a167bf4e7363c85d4af1bbff55141b3c493819af440cf8a4827833`。Isaac 的物理根为 trunk；修正后的虚拟 base 在 trunk 中为 `(0,+0.0000611544755962,-0.0201840868229783)` m。不能把 trunk、根刚体 COM 与几何 base 混称为同一点。

时间单位区别：一个并行环境是一个机器人及其仿真实例；一个回合是该机器人从重置到下次终止/超时的过程，最多 10 s，可包含多个滚动周期。PPO 一轮迭代每环境采集 32 个控制步（0.64 s），不是一个回合。startup 随机化在本次进程创建 4096 个环境时各自抽样，跨回合和训练迭代保持；新进程续训会重新创建并抽样。reset 随机化只在相应环境开始新回合时重新抽样。

## 3. 动作与观测

动作顺序严格为 FR hip、FL hip、RR hip、RL hip、FL thigh、RL thigh。PPO 输出先裁剪到 `[-1,1]`，再线性映射，环境还会裁剪最终位置目标：

| 关节 | 目标位置 rad | 目标范围 rad |
|---|---|---|
| 右 hip | `+0.52 + action` | `[-0.48,1.52]` |
| 左 hip | `-0.52 + action` | `[-1.52,0.48]` |
| 左 thigh | `2.065 + 0.385*action` | `[1.68,2.45]` |

两个左大腿独立动作；右大腿固定 1.68、所有 calf 固定 −2.82 rad。hip URDF 限位 ±1.588 rad；左大腿物理限位 1.68～2.45 rad。物理约束可能瞬时越界，不等于目标超限。没有额外动作滤波、延迟模型或显式动作变化率惩罚。

Actor/Critic 都是下列顺序的 35 维，无观测堆帧、无 RNN、无高度扫描或坡度角输入。Actor 加噪声，Critic 不加；网络不做经验归一化。

| 分量 | 维度 | 定义 | Actor 加性均匀噪声 |
|---|---:|---|---|
| base_lin_vel | 3 | 根合并刚体 COM 的机体系线速度 | ±0.1 m/s |
| base_ang_vel | 3 | 机体系角速度 | ±0.2 rad/s |
| projected_gravity | 3 | 机体系投影重力方向 | ±0.05 |
| velocity_commands | 2 | 机身 x 滚速、世界 z 偏航命令 | 无 |
| joint_pos | 6 | 上述顺序；减出生默认角，hip 为右 +1.52/左 −1.52，左 thigh 1.68 | ±0.01 rad |
| joint_vel | 6 | 相对默认速度（默认零），未额外缩放 | ±1.5 rad/s |
| actions | 6 | 上一步已裁剪动作 | 无 |
| rolling_cycle | 4 | 窗口平均滚速/10、窗口平均偏航、净负向滚角进度/(2π)、距上次整圈时间/3 | 无；整体裁剪 ±10 |
| support_contacts | 2 | 左、右任一指定刚体净接触力模长 >5 N | 无 |

前七项 scale=1、clip=±100。左接触集合 `(FL|RL)_(hip|thigh)`，右集合 `(FR|RR)_hip`；固定 calf/foot 合并进相应链。传感器自身的 1 N 参数不是这里 5 N 的策略接触门槛。地形扫描传感器仍可能存在于 scene，但 `height_scan=null`，不会提供给策略。

判断“继续滚还是启动”的信息来自角速度、重力、接触、周期统计及上一动作；没有显式两模式标签，也没有额外启动奖励。全辅助初态无法充分覆盖静止启动分布。

## 4. 命令与重置

- 滚速始终均匀采样 `[-14,-6] rad/s`，没有相反方向命令。滚速是机身 x 分量，负方向在初始朝向对应世界 +y 推进；不是世界某固定水平轴滚速。
- 偏航是世界 z 分量，均匀采样当前课程的 `[-yaw_max,+yaw_max]`。重采样间隔 1000 s，实际 10 s 回合内保持同一命令，重置时重新采样。
- 出生坡面相对 roll 为 `-2 ± phase_half_width`，环绕到 `[-π,π)`；世界姿态 roll 再加坡角；初始 pitch/yaw 为 0。
- 出生右 hip +1.52、左 hip −1.52、左 thigh 1.68 rad，关节速度零；不是父类旧的随机关节角/速度。
- 高度按碰撞网格离线表插值：721 个相位点（0.5°），3 mm 间隙，沿坡面法线放置。运行会检查高度表 URDF 哈希和默认关节角是否匹配；初始根高度不是所有相位固定 0.113 m。
- 初速系数为 `s` 时，坡面切向平移速度 `0.8*s m/s`，滚速 `-(0.8/0.112)*s = -7.142857*s rad/s`。这里 0.112 m 是代码的初速/课程半径常数，不是本次重新拟合的精确几何半径；初速也不是按每个滚速命令单独匹配。

## 5. 课程：目标范围与实际进度

| 课程 | 阶段（按索引升级） | 11999 次迭代实际值 |
|---|---|---|
| 初速辅助系数 | 1, 0.8, 0.6, 0.4, 0.2, 0 | **1，仍全辅助** |
| 相位半宽 rad | 0.10, 0.20, 0.35, 0.55, 0.80, 1.20, 1.80, π | **0.10** |
| 偏航上限 rad/s | 0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50 | **0** |
| 上坡上限 ° | 0, 2, 4, 6, 8, 10 | **0** |
| 下坡上限 ° | 0, 3, 6, 10, 15, 20, 25, 30 | **0** |

### 相位、偏航、辅助的联合升级

以下为 model_11999 的历史门槛；当前代码的单圈偏航误差门槛已改为 ≤0.06 rad/s，其余条件相同。

每个有效结束回合进入统计。回合成功要求实际时长 ≥6 s、完成 ≥4 个净负向整圈；至少 80% 完整圈滚速相对误差 ≤20%，偏航成功另外要求至少 80% 完整圈偏航误差 ≤0.15 rad/s。这里分别统计滚速好圈比例和偏航好圈比例，**不是要求每一个好圈同时满足两项**。

累计至少 2048 回合且距上次评估 ≥500 控制步，滚速成功回合率 ≥75%、滚速和偏航都成功的回合率 ≥70% 才算一批通过；连续两批通过后升级一项。顺序为辅助→相位→偏航循环，达到终点的项跳过。失败清除连续通过计数，保持等级；没有按迭代数自动升级或降级。新抽样在重置时应用。此联合统计没有给回合保存相位/偏航/辅助等级标签，升级后仍在运行的旧难度回合可能进入下一批；坡度统计另有等级过滤。

### 坡度课程

真实碰撞网格覆盖 −30～+10°，1° 间隔，共 41 块，每块宽 40 m、长 60 m，无水平底平面覆盖下坡。出生点在坡内分散，机器人跨环境碰撞过滤。初始世界 +y 为前进方向，正坡上坡、负坡下坡；转弯后实际顺坡分量改变。

每回合 20% 平地、40% 上坡类别、40% 下坡类别；类别内均匀抽样整数坡度 0～当前上限，故第 0 级所有环境都在平地。上下坡独立统计，只计本等级且坡度 ≥ceil(当前上限×0.75) 的前沿回合；第 0 级统计各自类别的平地回合。

每方向至少 512 个前沿回合、间隔 ≥500 控制步、连续两批 75% 滚速/70% 联合成功才升级。另需沿初始坡面 +y 切向净位移 ≥`0.25*0.112*abs(cmd_roll)*实际回合时长`，防止后滑空转通过。这是课程门槛，**不是平移速度奖励**。坡度与上述联合课程分别推进，但都要求偏航达标。

`rolling_curriculum.json`、`slope_curriculum.json` 只在升级时写出，缺文件不能单独证明异常或等级为零。通用 PPO checkpoint **不序列化环境课程等级和统计批次**；续训会重建课程。此次切换时旧等级恰好全为零，难度一致，回合和未完成统计重新开始。后续从非零等级续训必须显式恢复课程，不能仅加载 PPO 权重。

## 6. model_11999 历史实验的全部奖励（当前默认变更见页首）

最终 `self.rewards = LeftThighCycleRewardsCfg()` 替换父类奖励对象，**只有下面 5 项**。父类文件中仍能搜到的存活、横向速度、关节力矩平方、关节加速度平方、关节镜像、动作变化率、旧 Z 速度/防跳/高度项均不在本实验奖励表里。

令 `T=2π/abs(cmd_roll)`，约 0.449～1.047 s。`mean_roll`、`mean_yaw` 为最近目标周期长度的有符号滑动均值，非整数步窗口线性插值，开头不足窗口使用已有样本；这与用于课程的真实完整圈统计不同。

```text
roll_score = exp(-((mean_roll-cmd_roll)/1.5)^2)
yaw_score  = exp(-((mean_yaw-cmd_yaw)/0.25)^2)
reward_step = 0.02 * [
    8 * roll_score
  + 5 * roll_score * yaw_score
  - 1e-4 * (a_world.x² + a_world.y² + 2*a_world.z²)
  - 2e-6 * (alpha_world.x² + alpha_world.y² + 2*alpha_world.z²)
  - 1e-4 * sum(abs(applied_joint_torque * joint_velocity))
]
```

| 奖励项 | 权重 | 重要定义 |
|---|---:|---|
| roll_cycle_tracking | +8 | 滚速窗口均值跟踪 |
| yaw_cycle_tracking | +5 | 乘滚速得分，停车不能单靠零偏航拿高分 |
| base_linear_acceleration | −1e-4 | 新几何 base 点、世界三轴、Z 系数 2 |
| base_angular_acceleration | −2e-6 | 世界三轴角加速度、Z 系数 2 |
| joint_power | −1e-4 | 六个活动关节绝对机械功率之和，不是电池能耗 |

base 速度由 `root_link_lin_vel_w + omega_world × (R_world_trunk * offset_trunk_to_base)` 得到，不能把根 COM 速度代入 link 原点速度。加速度在世界系按 20 ms 差分；重置后首帧为零，每控制步只更新一次，不计跨回合跳变。世界 Z 始终竖直，包括坡面；这是真实运动学加速度，自由落体包含 −9.81 m/s²，未减重力，也不是 IMU 比力。角加速度与同刚体上的参考点平移无关。

角/线加速度正则不做周期平均，但 20 ms 差分仍不能完整捕捉 5 ms 接触尖峰。周期均值允许蹬地/滑行交替，也可能让窗口内正负偏航波动抵消。奖励无“停止/启动”额外门控。满跟踪时正奖励上限每步 0.26，未计惩罚。日志 `Episode_Reward/*` 按回合累计后除最大回合时长 10 s，不是某一时刻直接代入公式的每步奖励。

## 7. 终止、随机化与容易遗漏的难度

| 结束条件 | 类型 |
|---|---|
| 10 s 最大回合时长 | timeout，算法按超时处理 |
| 距上次完成净负向整圈超过 3 s | 失败终止；静止启动也必须及时完成首圈 |
| trunk 原点到坡面的法向高度 >0.5 m 或 <−0.05 m | 失败终止；参考点不是新 base |
| 机身 x 轴在坡面法向上的投影绝对值 >0.7 | 失败终止 |
| 相对坡块中心 abs(x)>19 m 或 abs(y)>29 m | timeout，提前离开边界 |

非法全身接触终止关闭；滚动本就需要壳体触地。没有额外终止惩罚项。训练启动 `init_at_random_ep_len=True`，会打散首批 episode_length_buf；课程自身实际回合计时另行维护，因此首批提前 timeout 不满足 6 s 时不会被算成成功。

以下表格为历史 model_11999 配置；当前质量、增益与 COM 范围见页首更新。

| 随机化 | 分布 | 时机 |
|---|---|---|
| 各刚体接触材质 | 静摩擦 0.3～1.0、动摩擦 0.3～0.8、恢复系数 0～0.5；64 桶 | startup |
| trunk 质量 | 加 −3～+3 kg，重算惯量 | startup |
| 其他合并刚体质量 | 乘 0.5～1.5，重算惯量 | startup |
| trunk COM 偏移 | x/y/z 各 −0.10～+0.10 m | startup |
| 六个关节 Kp/Kd | 各乘均匀 0.5～2.0，即 Kp 10～40、Kd 0.25～1.0 | 每次 reset |
| 观测噪声 | 第 3 节表 | Actor 观测 |
| 相位、命令、坡度 | 第 4/5 节 | reset |

地面静/动摩擦各 1.0、恢复系数 0，混合方式均为 multiply；所以机器人恢复系数随机化不等于地面接触实际恢复系数 0～0.5，与该地面相乘为 0。startup 随机化不在每次回合重新抽样。外力/力矩扰动与定时推搡均关闭。

尤其应关注历史实验 ±0.10 m 的 COM 偏移和其他 body 0.5～1.5 倍质量，这是继承后实际生效的宽随机范围；它们会改变接触与驱动难度。后续配置已按用户要求把 COM 缩至 ±0.03 m、trunk 质量偏移缩至 ±1 kg、其他刚体质量及 Kp/Kd 倍率缩至 0.8～1.2。未据此认定课程停滞的单一原因，尚无新训练效果数据。

## 8. PPO、续训和动作饱和

| 项目 | 参数 |
|---|---|
| Actor/Critic | 各 512→256→128，ELU；独立网络，无观测归一化 |
| 动作分布 | Gaussian，配置初始 std=0.8；续训实际加载旧分布参数 |
| rollout | 每环境 32 步，即 0.64 s；每迭代 131072 transition |
| 优化 | Adam；每批 5 epochs、4 mini-batches |
| 学习率 | 配置 8e-4，adaptive schedule、desired KL=0.01；运行学习率可变，不能当固定值 |
| PPO clipping | 0.2；value loss coefficient=1，clipped value loss 开启 |
| 折扣 | gamma=0.99、GAE lambda=0.95 |
| 熵系数/梯度 | entropy_coef=0.008，max_grad_norm=1 |
| 保存 | 每 100 次迭代及训练结束 |
| 对称增强/RND | 均未配置 |

本阶段从旧坡度 `model_5600.pt` 恢复策略、价值网络、优化器，再运行 6400 次，最终编号 11999。原平地→坡度→新 base 奖励是连续训练历史，不是当前奖励从零训 12000 次。此阶段日志累计 838860800 transition、约 8 h 18 min；总步数是本阶段统计，不应冒充所有前序训练的累计量。

最终日志 `Mean action std=20.52`，远大于执行裁剪范围 ±1；这会使采样动作频繁落在边界，与“初始 std=0.8”并不矛盾。这里只确认日志值与裁剪配置，未在本次审计计算逐关节饱和比例。Gazebo 推理取确定性均值再裁剪，与有探索噪声、物理随机化的训练采样有差别；课程停滞可能涉及这种差异，尚需专项诊断。

后续读取 `model_11999` 确认：六个动作的 std 按动作顺序为 `[25.9322,21.8940,24.7600,0.08447,24.4844,25.9938]`；20.52 是平均值，不是每个关节都一样。训练动作是 `clip(mu + std*Normal(0,1), -1, 1)`，std 没有角度单位；位置映射在裁剪之后。五个高 std 关节即使 mu=0，理论上也至少约 96% 的采样超出 ±1（均值偏离 0 只会更高），这是分布推导，非轨迹测得的饱和比例。RL hip 的 std 很小。现有课程未记录每个评估批次的滚速/联合成功率，无法仅用终局等级判断具体卡在哪条门槛；能确认的是没有完成连续两批达标。降低初速辅助与相位/偏航升级一样，仍被联合偏航成功门槛约束，并非只由滚速表现决定。

## 9. 结果边界与复现入口

11999 最后一条日志：mean reward=58.35，mean episode length=451.98 控制步（约 9.04 s），周期滚速误差指标 1.3125 rad/s、偏航 0.1354 rad/s。日志指标不是独立评估集，也不是每条轨迹全程均值。末次所有课程等级仍为零，不能声称已训练到 ±0.5 rad/s 偏航、10° 上坡、30° 下坡或零初速。

Guide 的 [28 组迁移测试报告](/home/eureka/docker-20.04/unitree_guide/data/rl_roll_20260927/report/report.md) 支持平地滚速跟踪，偏航和下坡未通过。Guide 使用 Gazebo、完整 12 电机 PD 保持其余关节、固定物理参数与真值桥接；训练为 PhysX、6 自由度固定链、宽物理随机化。该测试不能冒充与训练完全相同的环境或实机验证。

原始 [启动脚本](../logs/rsl_rl/tqbot2_roll_left_thigh_slope/launch_new_base_from_slope5600.sh) 精确记录当时的续训命令和 CLI 覆盖，其中 `max_iterations=6400` 是本阶段追加次数，`debug_vis=false` 是训练可视化覆盖。需要重现实验时先核对来源 checkpoint、依赖版本、URDF/高度表与课程恢复逻辑；不要把运行旧脚本当成对已完成实验的无副作用查看。

主要代码入口：[坡度环境](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/config/quadruped/tqbot2/left_thigh_slope_env_cfg.py)、[左大腿动作和奖励](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/config/quadruped/tqbot2/left_thigh_cycle_env_cfg.py)、[命令/观测/重置/联合课程](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/hip_only_cycle.py)、[周期统计与升级规则](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/cycle_tracking.py)、[坡面运行逻辑](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/slopes.py)、[坡度阶段与网格](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/slope_tracking.py)、[加速度正则](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/base_acceleration.py)、[PPO 配置](../source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/config/quadruped/tqbot2/agents/rsl_rl_ppo_cfg.py)。
