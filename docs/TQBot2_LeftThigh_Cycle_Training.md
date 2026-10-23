# TQBot2：允许左大腿运动的周期滚动训练

> 当前新 base 坡度模型的完整参数、实际课程进度和代码同步检查，见 [当前训练配置总览](TQBot2_Current_Training_Config.md)。本页保留专项设计与训练历史。

2026-09-29 当前奖励改为两套历史奖励输出各占50%；窗口与偏航规则各自保留，三项正则不变。见 [平均奖励实验](TQBot2_Averaged_Reward_Training.md)。下文09-28公式为平均之前的历史版本。

2026-09-28 物理随机化更新：当前左大腿及坡度任务 Kp/Kd 各乘 0.8～1.2（Kp 16～24、Kd 0.4～0.6），trunk 质量加 −1～+1 kg，其他合并刚体质量乘 0.8～1.2；COM 每轴 ±0.03 m。采样时机不变，质量变化仍重算惯量。供下次启动/续训使用，历史模型不变，尚未启动新训练。

任务 ID：`RobotLab-Isaac-Roll-LeftThigh-Cycle-TQBot2-v0`。
配置：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/config/quadruped/tqbot2/left_thigh_cycle_env_cfg.py`。
日志目录：`logs/rsl_rl/tqbot2_roll_left_thigh_cycle`。

2026-09-27 配置更新：trunk 质心随机偏移改为 x/y/z 各 ±0.03 m，坡度子类同步继承。供后续新启动或续训使用，尚未重新启动；此前 model_11999 仍对应 ±0.10 m 的历史配置。

2026-09-28 奖励更新：两项跟踪奖励改为最近实际净负向滚过 2π 的时间均值，首圈前无跟踪奖励；偏航为 `5 * I(-mean_roll > 6) * exp(-((mean_yaw-cmd_yaw)/0.125)^2)`，不再乘滚速匹配得分。偏航尺度为 0.125 rad/s，误差 0.05 时保留 85.214%；当前单圈偏航成功门槛为 ≤0.06 rad/s（数值容差 1e-7），三项惩罚和回合/批次通过比例保持原值。仅奖励统计改变，观测中的目标周期均值保留，以兼容现有 35 维策略和 Guide。新默认尚未用于训练；下面旧公式描述 model_11999 的历史设置，当前算法与验证见总览页首。

在 hip-only 周期课程配置上增加左前、左后大腿自由度。动作顺序为 FR hip、FL hip、RR hip、RL hip、FL thigh、RL thigh；共 6 个动作，Actor/Critic 观测均为 35 维。维度改变，需要从零训练，不能直接加载原来的 4 动作 checkpoint。

## 关节与动作

PPO 动作裁剪到 `[-1,1]` 后，目标位置为：

| 关节 | 目标位置 rad | 目标范围 rad |
| --- | --- | --- |
| 左 hip | `-0.52 + action` | `[-1.52, 0.48]` |
| 右 hip | `+0.52 + action` | `[-0.48, 1.52]` |
| 左前、左后 thigh | `2.065 + 0.385 * action` | `[1.68, 2.45]` |

两个左大腿独立控制，出生角度仍为 1.68 rad；右大腿固定 1.68 rad，全部小腿固定 -2.82 rad。新 URDF 同时设置左大腿物理关节限位 1.68～2.45 rad。仿真限位由约束求解器执行，实际角度可能短暂小幅越界。左大腿沿用 hip 的执行器参数：力矩上限 23.7 Nm、速度上限 30.1 rad/s、刚度 20、阻尼 0.5，并继承增益随机化。

左侧接触观测包括 hip 和新的 thigh 刚体，避免拆开固定关节后漏掉外壳接触。出生高度表使用新 URDF、初始左大腿角度和碰撞几何重新生成。

## 命令、课程和奖励

滚速命令始终在 `[-14, -6] rad/s`，没有正向滚速命令。沿用 roll `-2.0 ± 0.10 rad` 起步的相位课程，偏航命令从 0 按学习表现扩大到 ±0.5 rad/s。

出生时的初始角速度和平移速度按同一个辅助系数缩放：

| 辅助系数 | 初始滚速 rad/s | 初始 world-y 速度 m/s |
| --- | --- | --- |
| 1.0 | -7.142857 | 0.80 |
| 0.8 | -5.714286 | 0.64 |
| 0.6 | -4.285714 | 0.48 |
| 0.4 | -2.857143 | 0.32 |
| 0.2 | -1.428571 | 0.16 |
| 0.0 | 0 | 0 |

每次升级只改变一项，顺序为“降低初速辅助 → 扩大相位范围 → 扩大偏航范围”，循环推进，已到终点的课程跳过。连续两批达标才升级，每批至少 2048 个回合、相隔至少 500 控制步，滚速成功率 ≥75%、滚速与偏航同时成功率 ≥70%。成功回合至少 6 秒、4 个实际净负向整圈，其中 ≥80% 整圈的滚速相对误差 ≤20%、偏航绝对误差 ≤0.06 rad/s（历史 model_11999 为 ≤0.15）。失败保持当前等级，不按训练时间自动削弱辅助。新等级在各环境下次重置时应用。最终所有新回合均零初始速度；仍需通过训练确认从静止启动能力。

TensorBoard 的 Curriculum 组记录 `assist_level`、`initial_speed_scale`、`initial_roll_rate` 和 `initial_linear_speed`。升级时记录 `rolling_curriculum.json`；与原任务一样，通用 PPO checkpoint 不自动恢复环境课程等级。相位、偏航阶段及其他设置见 `TQBot2_HipOnly_Cycle_Training.md`。无整圈超时仍为 3 秒，因此静止启动也需在这段时间内完成首圈。

滚速使用机身 x 轴角速度，偏航使用世界 z 轴角速度。两者都取最近 `T = 2*pi/abs(cmd_roll)` 秒的滑动平均，窗口约 0.449～1.047 秒；回合开头不足一个窗口时使用已有样本。

```text
roll_score = exp(-((mean_roll - cmd_roll) / 1.5)^2)
yaw_score  = exp(-((mean_yaw  - cmd_yaw)  / 0.25)^2)
v_base_world = v_trunk_link_world + omega_world cross (R_world_trunk * r_trunk_to_base)
a = (v_base_world_now - v_base_world_previous) / 0.02
alpha = (omega_world_now - omega_world_previous) / 0.02
joint_power = sum(abs(applied_joint_torque * joint_velocity))
reward_per_step = 0.02 * (8 * roll_score + 5 * roll_score * yaw_score
                          - 1e-4 * (a.x^2 + a.y^2 + 2*a.z^2)
                          - 2e-6 * (alpha.x^2 + alpha.y^2 + 2*alpha.z^2)
                          - 1e-4 * joint_power)
```

偏航奖励权重为 5，乘滚速得分，防止停车保持零偏航获得高分。滚速完全跟踪时，偏航平均误差 0、0.25、0.5 rad/s 分别对应加权偏航奖励 5、1.839397、0.091578（乘控制步长前）。停转后的平均滚速为零时，即使偏航完全匹配，在最慢命令 -6 rad/s 下该奖励也仅约 5.63e-7；更快命令下更低。奖励跟踪平均值，因此窗口内正负偏航波动可能相互抵消。没有横向速度奖励。

2026-09-24 更新：以新 base 的三轴线加速度、三轴角加速度正则取代独立滚动/偏航角加速度项。坐标统一为世界系，Z 为竖直方向，Z 权重为 X/Y 的 2 倍。线加速度系数为 (-1e-4, -1e-4, -2e-4)，角加速度系数为 (-2e-6, -2e-6, -4e-6)。旧版 TQBot2 的 `lin_vel_z_l2`、`roll_anti_jump`、`base_height_above` 已停用，当前课程配置不再引入这些项。终止条件仍保留。这里是运动学加速度，自由落体 a_z=-9.81；没有减去重力或使用 IMU 比力。

参考点来自 unitree_guide 的 `0211114`：trunk→base=(0, +0.0000611544755962, -0.0201840868229783) m。Isaac 仍以物理 trunk 为根，用 root_link_lin_vel_w 加上 omega×r 得到 base 点速度；不能用 root_lin_vel_w（质心速度）直接代替。差分在世界系完成，不对旋转机体系的分量直接差分。角加速度不随同一刚体上的参考点平移而变。每个控制步 20 ms 更新一次，重复读取不改变结果；每个环境重置后第一帧只建立历史、不计惩罚，以排除跨回合跳变。该估计是控制步平均加速度，不代表 5 ms 接触尖峰。

能耗项保留 `joint_power`，权重 -1e-4，统计全部六个运动关节，100 W 对应 -0.01（乘步长前）。它是绝对机械功率近似，正负关节功率不抵消，不含电机电损耗。速度跟踪仍取周期平均；加速度正则不做周期平均。

注意：此变更是后续启动/续训配置；已在运行的实验不会热更新奖励。动作和观测维度不变，已有 checkpoint 可以续训，但原有实验日志对应旧奖励。

PPO 参数沿用原配置。本次只增加左大腿控制能力，没有调整此前训练出现的动作饱和问题。

## 启动

```bash
cd /home/eureka/robot_lab
source /home/eureka/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
python -u scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --device cuda:0 \
  --task RobotLab-Isaac-Roll-LeftThigh-Cycle-TQBot2-v0 \
  --num_envs 4096 --max_iterations 12000 --seed 42 \
  env.commands.base_velocity.debug_vis=false
```

## 验证

- `python -m unittest scripts.tests.test_tqbot2_description`：5 项测试通过，包含六自由度、左大腿轴和原点保持不变、关节限位、固定关节及出生高度表验证。
- `python -m unittest scripts.tests.test_roll_cycle_tracking`：9 项测试通过，包含辅助逐级降至零、失败不升级、一次只改变一种难度、滚动和偏航角加速度的跨回合隔离，以及停转时偏航奖励接近零。
- Isaac 仿真：`python scripts/tests/tqbot2_roll_cycle_smoke.py --left_thighs --headless --device cuda:0`，32 个环境、256 步随机动作通过，检查动作裁剪、观测维度、五项奖励、偏航权重 5、接触刚体、命令范围、各辅助等级出生速度、零初速仿真和部分重置。
- 新配置仿真日志：`outputs/left_thigh_roll_check/yaw_penalty_smoke.log`。

## 新 base 正则验证（2026-09-24）

16 项周期/坡度/参考点运动学 CPU 测试与 5 项模型一致性测试通过。运动学测试覆盖匀速滚动中心不会因 trunk 绕转被误罚、偏心点向心加速度、世界竖直自由落体、重复读取及部分重置。64 环境坡度仿真通过，奖励项为滚速跟踪、偏航跟踪、新 base 线加速度、新 base 角加速度、关节机械功率；日志见 `outputs/base_acceleration_regularization_check/slope_smoke.log`。

32 环境平地左大腿课程仿真亦通过，覆盖 256 控制步、非零负惩罚、全部初速辅助等级与重置、动作和观测维度不变；日志见 `outputs/base_acceleration_regularization_check/flat_smoke.log`。

## 新配置续训已启动（2026-09-24 21:03）

原训练在 model_5600.pt 完整保存后停止。新实验 `2026-09-24_21-03-39_base_accel_world_z2_from_slope5600` 已从该检查点恢复策略、价值网络和优化器，以新 base 三轴线加速度/角加速度正则继续训练。4096 环境、seed 42，再训练 6400 次，总目标仍为 12000 次。已核对新配置快照、日志中两项新惩罚、5602 次迭代和新检查点；原两项单轴角加速度奖励不再启用。

相位、偏航、初速辅助、上坡与下坡等级在切换时均为 0，续训保持该难度；未完成的成功率统计批次和环境回合重新开始。启动脚本为 `logs/rsl_rl/tqbot2_roll_left_thigh_slope/launch_new_base_from_slope5600.sh`，控制台日志为同目录 `console_new_base_from_slope5600.log`，tmux 会话 `tqbot2_roll_left_thigh_slope_base_accel`。新实验目录的 `resume_source.json` 与 `training_source_snapshot/` 保存来源与正则源码。
