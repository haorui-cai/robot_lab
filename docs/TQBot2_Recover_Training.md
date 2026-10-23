# TQBot2 Recover：任意朝向恢复到趴姿

## 任务入口

- 训练：`RobotLab-Isaac-Recover-Flat-TQBot2-v0`
- 回放：`RobotLab-Isaac-Recover-Flat-TQBot2-Play-v0`
- 算法：RSL-RL PPO，日志目录 `logs/rsl_rl/tqbot2_recover_flat/`
- 配置：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/recover/config/tqbot2/recover_env_cfg.py`

用户于 2026-09-18 明确：关节摆法无所谓，主要看机身朝向。当前目标是机身正面朝上、
贴近地面且稳定，不跟踪固定腿部关节角、世界 yaw 或落点。2026-10-12 用户要求抑制恢复后的腿部抖动，因此新版本在保持阶段约束关节速度，并排除腿仍在机身上方的假成功。
`State_FixedLie` 的每腿 `[0, 1.36, -2.65]` rad 仅保留为动作零点、观测参考和部分初态，
不是奖励或成功条件；任意其他合法腿姿都可以完成 Recover。

## 模型同步与兼容

来源：`/home/eureka/docker-20.04/unitree_guide/src/unitree_ros/robots/TQBot2_description`。
同步 URDF、meshes、xacro、config、launch、package.xml 和 CMakeLists.txt，
`sync_manifest.json` 记录源路径与逐文件 SHA-256。原始 ROS URDF 保持源文件内容。

新版模型主要变化：髋关节范围 ±1.588 rad，大腿范围 [-3.141, 6.108] rad；
更新机身碰撞体的缩放和位置，增加半径 0.02 m 的膝部碰撞球及接触参考坐标系。
小腿范围仍为 [-2.975, -0.916] rad。更新后的共同资产也会被现有 Velocity/Roll 任务使用，
旧 checkpoint 对新动力学的适用性需要重新评估。

Isaac 使用生成的 `urdf/TQBot2_isaac.urdf`，有两处兼容处理：

1. 移除无质量/无碰撞的虚拟 `base` 和 `floating_base` 固定关节，以 `trunk` 为根。
   源模型的虚拟基座原点位于 trunk 下方 24.7 mm；训练始终采用 trunk 坐标。
   所有物理 link、惯量、碰撞体及其相对 trunk 的坐标均保持不变。
2. 将 `base_link.STL.dae` 引用改成内容相同的 `base_link.dae` 相对符号链接。
   Isaac Sim 5.1 使用文件 stem 生成 USD prim，带点的 `base_link.STL` 会触发 `Used null prim`。

重新同步：

```bash
python scripts/tools/sync_tqbot2_description.py
python scripts/tests/test_tqbot2_description.py
```

脚本保留目的目录中源工程没有的既有文件，不删除其他资产。

## 随机初始化与控制

所有回合以 180° roll、随机 yaw 从地面的四脚朝天状态开始，trunk 初始高度 0.096 m。
初始关节采用参考姿态，关节速度、根线速度和角速度为零；每次完整或部分重置均断言
投影重力 z > 0.99。地面摩擦/恢复系数在启动时随机。Play 使用同一初始化，关闭观测噪声。

12 维动作按 FR/FL/RR/RL、hip/thigh/calf 的顺序控制。
RSL-RL wrapper 将动作先裁剪至 [-1, 1]，再映射为参考零点的关节偏移，最后按 URDF 限位裁剪位置目标。
hip/thigh/calf 缩放分别为 1.588/4.75/1.735 rad，可覆盖新版活动范围。
小腿参与控制。仿真 200 Hz，策略 50 Hz，单回合最长 16 s，给恢复后的保持留出时间。

Recover 单独使用 hip/thigh PD 40/2、calf PD 60/3，保留源 URDF 对应的力矩/速度上限，
仅启用 FR–FL 与 RR–RL 两组跨腿自碰撞，求解器迭代设为 8/2。同腿内部、腿与躯干、
前后腿之间仍过滤碰撞。过滤关系在机器人生成时写入，早于环境复制及 PhysX 初始化。
原资产的公共 PD 默认值不受此覆盖影响。

Policy 观测共 42 维：投影重力 3、体坐标角速度 3、相对趴姿关节角 12、
关节速度 ×0.1 共 12、上一动作 12。Critic 额外接收基座线速度 3 和离地高度 1，共 46 维。
Policy 不需要外部定位或接触力输入。这个接口与源工程另一个 47 维 `RecoverPolicy` C++ 模块不同，
训练出的策略不能直接加载进该模块；部署时必须另行对齐观测、动作缩放、PD 和控制周期。

## 奖励与成功判据

当前 v3 奖励：直立程度（权重 3）、低机身且机身低速并且腿放低的联合奖励（6）、
趴地状态（4）、朝向正确时的腿部空间位置 shaping（4）。没有关节角匹配奖励。
膝部采用 calf link 原点，足端由 calf 原点加局部 `[0, 0, -0.153]` m 计算，
以每条腿 thigh link 原点作为关节支点，膝/足最高点不高于该支点 4 cm。
这个空间约束允许不同折腿姿态，排除旧 Gazebo 测试中腿整体在躯干上方的反例；
它是可复核的工程判据，仍须通过 Gazebo 画面确认实际动作，不能等同于用户验收。

必须同时满足以下条件并连续保持 **4 s（200 控制步）**，才能以 `success` 结束回合：

- 投影重力 z < -0.95；trunk 离地高度处于 (0.02, 0.18) m。
- 根线速度 <0.15 m/s，角速度 <0.5 rad/s。
- 四条腿的膝/足端满足上述空间约束，不跟踪固定关节角。
- 所有物理 body 净接触力的竖直分量总和 >30 N；整机求和避免把腿之间的内力误判为地面支撑。
- 12 个关节速度 RMS <0.25 rad/s，最大绝对速度 <0.75 rad/s。
- 回合已运行至少 1 s；失去任一条件，连续计数归零。回合重置也清空计数。

原有力矩（-1e-5）、原始动作变化（-0.003）及限位（-2）正则保留。
新增接近恢复姿态时的关节速度 Huber 损失均值（-0.3，delta=1 rad/s）、
关节加速度 Huber 损失均值（-2e-5，delta=100 rad/s²）、实际裁剪后的关节位置目标
变化率 Huber 损失均值（-0.01，delta=1 rad/s）。小误差为半平方，大误差线性增长。
初次平方惩罚预检被旧策略的大幅振荡主导（负回报超过 2000），故停止该预检并保留日志
`quiet_hold_squared_penalty_preflight_20261012.log`，正式微调采用 Huber 控制奖励尺度。
保持权重为 `upright**8 * legs_down * exp(-height_excess**2 / 0.08**2)`，
翻身过程中较弱，接近正确趴地时增强，不使用锁存状态或额外 actor 输入。
目标变化率历史在完整/部分重置后跳过第一步，避免把重置位置差当成抖动。

成功一次奖励 +50（权重 2500，dt=0.02），超过 gamma=0.99 时逐步姿态奖励
最高折扣和 34，避免故意延迟结束。越界、异常高度与 16 s 超时结束回合。
探索标准差下限从 0.03 降至 0.003，允许训练阶段的随机动作逐渐安静下来；
确定性回放和导出仍使用 Gaussian 均值。
TensorBoard 成功率现在使用 v3 判据，不能和旧判据成功率直接比较。

## 训练、回放和验证

```bash
conda activate isaaclab
export OMNI_KIT_ACCEPT_EULA=YES

python scripts/tests/tqbot2_recover_smoke.py --headless

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Recover-Flat-TQBot2-v0 \
  --num_envs 4096 --headless --run_name recover_orientation_v2

python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Recover-Flat-TQBot2-Play-v0 \
  --num_envs 16 --checkpoint /absolute/path/to/model_XXXX.pt
```

默认 PPO 12000 iterations，每 100 次保存；actor/critic 均为 [256, 256, 128]。
当前 agent 使用本机 RSL-RL 5 的模型/分布接口，探索标准差通过 sigmoid 参数化限制在
[0.003, 0.6]，初值 0.35；关闭额外熵奖励。PPO 的采样、log-prob、熵和 KL
统一使用此有界标准差，确定性回放/导出仍使用高斯均值。
新版从头训练，避免继承旧版异常噪声和旧目标；旧 checkpoint 的分布参数与新版不兼容。
GPU 测试包含成功条件反例、连续保持和重置逻辑、URDF 导入、关节顺序、
42/46 维观测、随机动作 rollout，以及零动作下目标趴姿的物理可达性。
这些测试验证实现与目标姿态，不能证明训练后能够从任意姿态恢复；
后者需要收敛 checkpoint、分姿态/摩擦的恢复率统计与失败案例检查。

## 第一版验证记录（2026-09-18，保留历史，目标已被用户修正）

- 原始网格名带双后缀导致 Isaac 导入失败；按日志定位非法 USD prim 后采用网格别名修复。
- 虚拟 base 的移除用于保留现有 trunk 路径；它不是上述非法 prim 的直接原因。
- 沿用公共 PD 20/0.5 时，零动作趴姿约有 0.22 rad 最大误差；25/1.5、35/2 仍约有
  0.17–0.20 rad 误差，不能可靠达到成功阈值。保留这些失败记录，采用更高刚度后复测。
- 实际测试与训练日志在 `outputs/recover_validation/`；训练产物在上述日志目录。
- 最终验证：3 项 CPU 描述一致性测试、Ruff 静态检查/格式检查通过；固定 seed=42 的 GPU
  测试完成 200 步随机动作与 200 步趴姿保持。16/16 环境满足趴姿判据，最终高度约
  0.0767–0.0788 m，最大关节误差约 0.120–0.137 rad。
- 完整 PPO 短训练：128 环境 × 3 iterations 成功，保存 `model_0.pt`、`model_2.pt`。
  Play 加载该 checkpoint 后完成 64 步无渲染推理，并导出 TorchScript/ONNX。
  开启视频的首次回放停在渲染初始化，已结束该测试；本轮未验证视频输出。
- 正式训练使用 4096 环境、seed=42、12000 iterations，run name 为 `recover_prone`。
  进程和命令记录在 `outputs/recover_validation/training_process.json`，stdout 为
  `outputs/recover_validation/train_recover.log`。本次交付不声称策略已收敛。

## 朝向优先修订（2026-09-18）

- 问题：首版把 FixedLie 关节角误当成必要目标。用户确认腿部摆法不限后，删除关节角匹配
  奖励、联合奖励中的关节误差/速度项，以及成功条件中的关节角/关节速度检查。
- 原训练在约 3000 iterations 时成功率仍为零，动作 std 超过 80。旧 run 已停止，
  checkpoint、stdout 和 `training_process_v1.json` 原样保留；不把朝向奖励高等同于成功恢复。
- 新方案：以有方向性的投影重力为主要目标，保留触地、低机身和机身稳定要求；
  将原关节角奖励权重移给朝向奖励，成功奖金仍为 +30，连续保持仍为 0.5 s。
- 为抑制已观察到的探索噪声膨胀，增加本项目内的有界高斯分布并关闭熵奖励，
  不修改外部 RSL-RL 安装。CPU 测试检查极端参数下的噪声范围、梯度、log-prob/KL 和导出均值。
- 更新 GPU 测试：不同关节角、较大关节速度只要机身状态相同，奖励和成功判定都一致。
- 本轮验证通过：2 项分布 CPU 测试，关节无关性/连续成功 GPU 检查，16 环境随机 rollout，
  128 环境 × 5 iterations 的完整 PPO 训练，以及新版 checkpoint 的 64 步回放和
  TorchScript/ONNX 导出。短训练的动作 std 保持约 0.35；长期恢复率仍需评估。
- 新运行 stdout 为 `outputs/recover_validation/train_recover_orientation_v2.log`；
  当前进程记录仍使用 `outputs/recover_validation/training_process.json`。

## Gazebo 验证失败与判据更正（2026-10-12）

对 `2026-10-11_22-57-36_supine_cross_leg_20261011/model_700.pt` 运行八个偏航角的
四脚朝天初态测试。NumPy 与 PyTorch actor 推理在 100 组输入上一致；使用独立 ROS
11411 / Gazebo 11445，FR–FL、RR–RL 碰撞开启。测试数据位于
`outputs/recover_validation/gazebo_model700_report.json` 和同目录的轨迹 CSV。

**撤回成功结论。** 原数字统计只确认机身坐标系的 z 轴朝上、低机身和低速度，
没有确认用户要求的实际趴姿。用户观察八次均失败，最终画面独立检查也显示腿仍在
机身上方。Gazebo 模型姿态与 IMU 一致，因此此次误报不是 IMU 方向读取反号，
而是成功判据不足。最终 FR 大腿约 π rad，小腿原点高约 0.220 m，机身根高约 0.041 m。
8/8 数字朝向通过、5/8 数字稳定通过均不得再作为实际 Recover 成功率引用。

训练也使用机身投影重力、高度和速度判据，所以 Isaac 的数字成功率不能独立证明
策略已完成用户目标。后续应对照用户认可的真实趴姿检查物理朝向/支撑关系，
保留“不约束固定关节角”的要求，并对新的判据复核训练与 Gazebo。

测试模型局部替换 IMU 插件以避开传感器插件处理控制器服务时的物理线程死锁；
旧模型 IMU 插件的角速度坐标转换在桥接脚本中校正。此测试不声称完成地面接触力验证。

用户指出原 Gazebo 测试在机身落地前立即启动策略，而且翻转后腿部抖动。
测试桥接已改为先保持初态，确认机身与地面接触且根线速度 <0.08 m/s、角速度 <0.5 rad/s
连续一秒后才允许 actor 推理。原力话题一直返回零，实验 SDF 增加 ROS bumper 接触传感器
覆盖机身所有碰撞体，并只统计 ground_plane 接触。底座到 trunk 偏移直接从 robot_description 读取。
不再在机身数字判据达到后提前结束；持续观察 17 秒，记录关节速度与位置目标变化。

单次四脚朝天、yaw=0 验证：等待 1.18 秒落地后启动；启动时投影重力 z=0.99885，
背壳接触测量非零。最后五秒关节最大速度的中位数约 0.964 rad/s、峰值 1.174 rad/s；
相邻策略位置目标的最大变化中位数约 0.0093 rad、峰值 0.0635 rad。因此只确认落地门控已工作，
**不声称腿部抖动已修复，也不声称实际趴姿恢复通过**。数据见
`outputs/recover_validation/gazebo_model700_grounded_report.json` 和对应轨迹 CSV。

## 恢复后安静保持微调（2026-10-12）

旧 run `2026-10-11_22-57-36_supine_cross_leg_20261011` 已停止，所有 checkpoint 保留。
微调从其最新完整保存的 `model_10300.pt` 初始化 actor/critic，重置 optimizer 与迭代计数，
采用新判据、新奖励和较低探索噪声下限；42/46 维观测、12 维动作、PD、初态和碰撞设置不变。
由于分布下限改变，加载相同 noise_logit 后探索 std 会降低，actor 均值权重不变。
训练运行名 `supine_quiet_hold_v3_20261012`，4096 环境，seed=42，先训练 3000 iterations，
每 100 iterations 保存。`--warm_start` 与 `--resume` 互斥，前者不载入旧 optimizer/迭代数。
本次将学习率设为 1e-4、固定 schedule，避免新奖励的价值损失突然改变 actor 太多。

验证日志：`outputs/recover_validation/hold_smoke_20261012.log`。
旧策略新判据基线：`outputs/recover_validation/hold_baseline_model10300_20261012.json`，
评估使用 64 个四脚朝天初态、固定 seed=2026、16 秒连续执行、不因成功提前停止，
报告连续 4 秒安静恢复率以及最后 4 秒趴地占比、关节 RMS 速度、目标 RMS 变化率。
新判据包含腿在上方、自碰撞内力、噪声打断、完整/部分重置等反例回归。
训练启动及 checkpoint 来源记录在 `outputs/recover_validation/train_quiet_hold_v3_20261012.json`。
训练启动不表示抖动已修复；最终仍需要确定性回放和 Gazebo 落地后恢复的独立验证。

旧 model_10300 基线：64/64 最终机身朝上，最后 4 秒趴地几何判据占比约 94.55%，
但连续 4 秒安静恢复 **0/64**。最后 4 秒平均关节 RMS 速度约 17.03 rad/s，
目标 RMS 变化率约 179.97 rad/s。数值为 Isaac 本次独立评估，不能外推为 Gazebo 表现。
真实物理保持检查：16/16 环境固定零动作的参考趴姿可连续安静保持 4 秒，
用于证明新阈值可达，不是恢复策略成功率。

正式 Huber 微调日志目录：`/home/eureka/robot_lab/logs/rsl_rl/tqbot2_recover_flat/2026-10-12_09-16-06_supine_quiet_hold_v3_20261012`。
已确认成功加载 warm-start 权重、完成 PPO rollout/update 并保存新 `model_0.pt`；
训练初期回报及损失有限，不据此声称已经改善恢复或抖动表现。
