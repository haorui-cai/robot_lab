# TQBot2 左大腿滚动：上下坡课程

> 当前新 base 坡度模型的完整参数、实际课程进度和代码同步检查，见 [当前训练配置总览](TQBot2_Current_Training_Config.md)。本页保留专项设计与训练历史。

2026-09-29 当前奖励改为两套历史奖励输出各占50%；窗口与偏航规则各自保留，三项正则不变。见 [平均奖励实验](TQBot2_Averaged_Reward_Training.md)。下文09-28公式为平均之前的历史版本。

2026-09-28 物理随机化更新：当前左大腿及坡度任务 Kp/Kd 各乘 0.8～1.2（Kp 16～24、Kd 0.4～0.6），trunk 质量加 −1～+1 kg，其他合并刚体质量乘 0.8～1.2；COM 每轴 ±0.03 m。采样时机不变，质量变化仍重算惯量。供下次启动/续训使用，历史模型不变，尚未启动新训练。

任务：`RobotLab-Isaac-Roll-LeftThigh-Slope-TQBot2-v0`。
配置：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/config/quadruped/tqbot2/left_thigh_slope_env_cfg.py`。
实验目录：`logs/rsl_rl/tqbot2_roll_left_thigh_slope`。

继承左大腿周期滚动策略：6 个动作、35 维观测，左大腿 1.68～2.45 rad，滚速命令 -14～-6 rad/s。滚速奖励权重 8、偏航奖励权重 5、新 base 的世界系三轴线加速度/角加速度正则（Z 系数为 X/Y 的 2 倍）及机械功率惩罚。2026-09-28 当前默认：跟踪奖励取最近实际净负向一周的时间均值；仅当该圈目标方向速度幅值 >6 rad/s 才发偏航跟踪奖励，不再乘滚速跟踪得分；首圈前无两项跟踪奖励。偏航指数尺度为 0.125 rad/s，误差 0.05 保留 85.214%；单圈成功门槛收紧为 ≤0.06 rad/s。历史 model_11999 仍采用目标周期窗口及相乘公式，尚未以新奖励续训。动作和观测含义、网络尺寸不变，可接平地 checkpoint 续训，包含优化器状态。

## 坡面与出生状态

使用真实碰撞网格坡面，按 1° 分辨率覆盖 -30°～+10°。正向行进为世界 +y；正坡角为上坡，负坡角为下坡。偏航转弯后，上下坡分量随实际行进方向变化。

坡面方程为 `z - z0 = tan(theta) * (y - y0)`，每块宽 40 m、长 60 m，无水平底平面遮挡下坡。环境之间的机器人碰撞按 IsaacLab 环境分组隔离，同一坡面的出生点分散布置，避免 4096 个机器人集中在一个点。

出生 roll 在坡面坐标中按原相位课程采样，再叠加坡角。碰撞几何高度表沿坡面法线放置，初始平移速度沿坡面切向，初始角速度仍为负机身 x 轴滚速。初速辅助继续按原课程从 100% 降至零。无效高度使用机器人到坡面的法向距离，而非绝对世界 z；到达地块边缘前按超时重置。

## 坡度课程

初级均为平地。每回合按 20% 平地保留、40% 上坡课程、40% 下坡课程抽样；上下坡类别内，在 0 到当前上限间均匀采样整数坡度。两个方向独立升级：

| 方向 | 坡度上限阶段 |
| --- | --- |
| 上坡 | 0° → 2° → 4° → 6° → 8° → 10° |
| 下坡 | 0° → 3° → 6° → 10° → 15° → 20° → 25° → 30° |

只用当前等级、且坡度不低于当前上限 75% 的回合评估升级，避免大量容易坡度掩盖最难坡度的失败。0° 阶段使用各自类别的平地回合。每方向每批至少 512 个有效回合，评估间隔至少 500 个控制步；连续两批滚速成功率 ≥75%、滚速与偏航同时成功率 ≥70%，才增加一级。不随训练时间自动加坡。

成功条件沿用原课程：至少 6 秒、4 个净负方向整圈，≥80% 整圈滚速相对误差 ≤20%、偏航误差 ≤0.06 rad/s（浮点容差 1e-7；历史 model_11999 为 ≤0.15）。此外，净前进距离至少达到 `0.25 * 0.112 * abs(cmd_roll) * episode_time`，防止一边后滑一边空转也被判定为爬坡成功。这是课程门槛，不新增横向速度奖励。

坡度课程与原相位、偏航、初速辅助课程分别按表现评估。新坡度只在环境下次重置时应用；旧等级未结束的回合不混入新坡度等级的评估。TensorBoard 记录 `Curriculum/slopes/uphill_max_deg`、`downhill_max_deg` 及两方向等级，升级时写入 `slope_curriculum.json`。

与原任务一样，通用 PPO checkpoint 不自动恢复环境课程等级；本次从仍处于第 0 级的平地模型转入，因此所有课程从第 0 级开始。坡度课程的最终范围不代表策略已经具备相应能力。

## 录像与验证

平地录像使用用户回放中的 `model_3700.pt`，约 30 秒、960×540、50 fps，保留绿色命令箭头与蓝色实际速度箭头：
`outputs/tqbot2_roll_left_thigh_model3700/videos/play/rl-video-step-0.mp4`。

- 12 项 CPU 测试通过：原周期统计、奖励约束、辅助课程、坡度独立升级、极限范围、网格坡角与法向。
- `scripts/tests/tqbot2_roll_slope_smoke.py`：64 个环境，验证 +10°/-30° 实际接触、坡面相对出生高度、切向初速、坡面坐标相位、完整坡度抽样、有限奖励与观测、部分重置隔离。日志 `outputs/left_thigh_slope_check/smoke.log`。
- 平地回归检查：`outputs/left_thigh_slope_check/flat_regression.log`。

启动脚本与来源 checkpoint 清单在坡度实验目录中；原平地实验和 checkpoint 保留。

## 本次续训

2026-09-24 19:39 启动，来源为平地实验 `2026-09-24_17-12-26_left_thigh_yaw5_assist_decay/model_4600.pt`，先复制至坡度目录的 `bootstrap_flat` 并校验可加载，再停止旧平地进程。

新实验：`2026-09-24_19-39-53_up10_down30_from_flat4600`；4096 个环境，再训练 7400 次，使总迭代编号达到 11999（总目标 12000 次）。已确认模型恢复、新实验 checkpoint、两项课程和五项奖励正常运行。启动脚本：`logs/rsl_rl/tqbot2_roll_left_thigh_slope/launch_slope_from_flat.sh`；控制台日志：`logs/rsl_rl/tqbot2_roll_left_thigh_slope/console_from_flat4600.log`。

## 2026-09-24 新 base 正则配置更新

现配置采用 `base_linear_acceleration`（-1e-4 × [ax²+ay²+2az²]）和 `base_angular_acceleration`（-2e-6 × [αx²+αy²+2αz²]），替代原两个单轴角加速度项；计费前再乘 0.02 s。参考点与重置处理详见 `TQBot2_LeftThigh_Cycle_Training.md`。原运行实验 `2026-09-24_19-39-53_up10_down30_from_flat4600` 仍使用启动时的旧奖励；本次没有自动重启训练。

## 新配置续训已启动（2026-09-24 21:03）

原训练在 model_5600.pt 完整保存后停止。新实验 `2026-09-24_21-03-39_base_accel_world_z2_from_slope5600` 已从该检查点恢复策略、价值网络和优化器，以新 base 三轴线加速度/角加速度正则继续训练。4096 环境、seed 42，再训练 6400 次，总目标仍为 12000 次。已核对新配置快照、日志中两项新惩罚、5602 次迭代和新检查点；原两项单轴角加速度奖励不再启用。

相位、偏航、初速辅助、上坡与下坡等级在切换时均为 0，续训保持该难度；未完成的成功率统计批次和环境回合重新开始。启动脚本为 `logs/rsl_rl/tqbot2_roll_left_thigh_slope/launch_new_base_from_slope5600.sh`，控制台日志为同目录 `console_new_base_from_slope5600.log`，tmux 会话 `tqbot2_roll_left_thigh_slope_base_accel`。新实验目录的 `resume_source.json` 与 `training_source_snapshot/` 保存来源与正则源码。
