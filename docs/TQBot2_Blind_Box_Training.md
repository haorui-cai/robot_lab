# TQBot2 盲爬箱实验

任务：`RobotLab-Isaac-Climb-BlindBox-TQBot2-v0`。

这是独立、人工选择的爬箱技能。第一版假设操作者将机器人朝向平台后调用此策略；自动选择技能和退出切换尚未实现。

## 输入和部署边界

Actor 每帧有 57 个数，按 observation term 分组、每组从旧到新堆叠 12 帧，共 684 维：机身角速度、重力方向、爬箱命令、关节位置偏差、关节速度、上一动作，以及目标关节位置减实际关节位置。当前只训练固定的 `(0.4, 0, 0)` 命令，没有训练其他速度指令或自主退出。历史在 50 Hz 下覆盖 0.24 秒。初始阶段的历史重复首帧，部分环境 reset 会清除其旧历史。

Actor 不读取地形高度、箱子距离、足端接触标签、世界位置、episode 时间或仿真机身线速度。Critic 单帧输入包含机身线速度、高度采样和精确箱子几何，这些只用于训练。脚尖位置和接触力只用于奖励、成功判据和统计。Actor 使用的目标关节位置反馈应在实机由相同控制器保存；不能用仿真接触信息替代。

使用原关节位置动作缩放：hip 0.125，其余 0.25；加默认位置偏置后，目标裁剪到 URDF 关节范围的中心 90%。驱动力矩、速度上限和机器人模型保持现有配置。

## 地形、奖励和课程

地形为真实硬碰撞的宽平台库，平台高度 4–30 cm、每级 2 cm，顶部长度随机取 1.2、1.6、2.0 m。每个共享平台宽 16 m，机器人分布在不同横向位置；这是验证竖直越障能力的第一阶段，不是窄箱落脚实验。平台前缘距离地形原点 1 m；每次 reset 在原点前后随机，得到约 0.65–1.15 m 的机身到前缘距离。横向偏离起始路线超过 0.65 m 判失败，抑制绕行。所有高度只是实验范围，不代表机器人能够完成。

从 4 cm 开始，当前最高级累计 128 个完整 episode 的成功率至少 80%，连续两个不重叠批次通过才升级。25% episode 从已解锁等级回放，其余训练最高级。旧等级不推动更高等级升级。显式手动 reset 和固定高度评估不计入课程。

主奖励包括有正负号的目标距离变化、接近平台中心、足端顶部支撑和成功事件；没有逐时刻前进速度跟踪或预设 trot。关闭足高、同步步态、关节镜像等奖励，减少默认关节姿态、竖直速度和俯仰角速度惩罚。保留关节限位、动作平滑、驱动负担和机身接触惩罚。

成功要求四个真正足端都已在箱顶落地，每脚最近一次有效顶部接触不超过 0.6 秒；四个足端的水平位置位于箱顶有效范围，机身姿态/速度/高度合理，连续保持 0.5 秒，完成时至少两脚支撑。允许正常行走中的摆腿，水平速度上限 1 m/s、竖直速度上限 0.3 m/s，身体角速度范数上限 1.2 rad/s；这里的阈值只是第一版实验判据。各足从未在箱顶落地、身体倒下或跳过平台都不能算成功。

停稳独立统计，要求四足当前都接近箱顶、至少三脚支撑，机身线速度小于 0.25 m/s、角速度范数小于 0.6 rad/s，并连续保持 0.5 秒；它不是高度课程的升级门槛。第一版没有自主停止/技能切换模块。

足端位置由 URDF 中固定足端偏移和 calf 刚体姿态计算，不能直接使用位于膝部的 calf 原点。奖励管理器会乘以 dt，因此成功和失败的权重 +250/-250，在 50 Hz 下对应一次 +5/-5。失败代价用于抑制向目标扑倒、只获得短期前进奖励的行为。

`box_curriculum.json` 保存课程状态。常规 PPO checkpoint 不自动保存此状态；继续课程时显式设置 `env.box_curriculum_resume_path` 为之前运行的该 JSON 文件，或在实验脚本中传 `--curriculum_state`。

支持从当前 45 维、无观测归一化的 TQBot2 行走 MLP 初始化 actor。第一层扩展到 684 维，仅将旧权重放在六项旧观测的最新一帧位置，其他历史与新增跟踪误差的权重初始化为零；隐藏层和动作分布参数复制。三层隐藏维度保持旧 actor 的 512/256/128；新 critic、PPO optimizer 从零初始化。这样迁移前后确定性动作保持一致。

微调只训练第一层新增的历史/跟踪误差列，原输入列、偏置和后续隐藏层保持冻结；通过梯度掩码和清除对应 Adam 动量保证原参数不漂移。动作探索标准差设为 0.2 并固定，学习率固定为 1e-4。保存的模型仍是普通 MLP，不需要运行时残差网络。原网络参数不变并不保证总体策略不会退化，因此必须比较微调前后实测成功率。

归一化结构、输入维度或层结构不匹配会拒绝迁移。原 checkpoint 单独复制到输出目录，保存 SHA-256。`--checkpoint` 自动恢复网络尺寸/归一化结构，并读取同目录报告中的 `history_adapter_only` 设置；报告不在同目录时可显式传 `--history_adapter_only`。课程状态仍需通过 `--curriculum_state` 单独恢复。

## 执行

```bash
conda activate isaaclab
python scripts/tests/test_blind_box_geometry.py
python scripts/reinforcement_learning/climb/experiment.py \
  --headless --device cuda:0 --num_envs 256 --iterations 300 \
  --output outputs/blind_box_trial_20261012
```

实验脚本先验证 actor 输入隔离、有限物理状态、目标关节限位、部分历史重置，以及已知地面/箱顶站立的物理对照，然后在最低评估高度测零动作基线，训练 PPO，最后在固定 4/8/12/16 cm 高度评估。评估用确定性动作，保留传感器噪声和物理随机化，起始距离、姿态和平台长度随机；每个高度的首批 reset 使用固定种子，迁移前后可以比较同一批起始条件。`report.json` 分别报告成功、失败和超时，以及曾有两脚/四脚位于箱顶、四脚位于箱顶并有至少三脚支撑的次数。后几个统计不要求停稳，不能替代严格成功率。

传入 `--walking_checkpoint /absolute/path/model.pt` 时，会先测迁移后的原行走 actor 基线，再训练和评估。零动作基线不是现有行走策略对照。

已测过同一模型、种子、环境数量和评估案例时，可以用 `--baseline_report /absolute/path/report.json` 复用基线；脚本检查 checkpoint SHA-256 和三份任务定义的 SHA-256，并在新报告中保留基线来源。此选项不会伪造一次新的评估。

只检查环境和成功判据：

```bash
python scripts/reinforcement_learning/climb/experiment.py \
  --headless --device cuda:0 --num_envs 8 --validation_only \
  --output outputs/blind_box_validation
```

完整训练也可使用仓库入口：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Climb-BlindBox-TQBot2-v0 \
  --headless --device cuda:0 --num_envs 1024 --max_iterations 10000
```

从已有行走 actor 试训：

```bash
python scripts/reinforcement_learning/climb/experiment.py \
  --headless --device cuda:0 --num_envs 128 --iterations 100 \
  --walking_checkpoint /absolute/path/to/walking/model.pt \
  --trials_per_env 1 --eval_heights 0.04 0.08 0.12 0.16 \
  --output outputs/blind_box_warm_trial
```

只评估已有爬箱 checkpoint（自动恢复 actor/critic 的隐藏维度与归一化结构）：

```bash
python scripts/reinforcement_learning/climb/experiment.py \
  --headless --device cuda:0 --num_envs 64 --iterations 0 \
  --checkpoint /absolute/path/model_299.pt \
  --trials_per_env 4 --eval_heights 0.04 0.08 0.12 0.16 \
  --output outputs/blind_box_evaluation
```

短训练用于检查训练链路和低平台学习趋势，不能证明高箱子泛化、窄箱落脚或实机可部署。下一步应加入实际窄箱碰撞几何、不同接近距离/未见平台尺寸的分层评估，并对历史、奖励、驱动饱和做消融。
