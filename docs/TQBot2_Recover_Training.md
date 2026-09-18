# TQBot2 Recover：任意朝向恢复到趴姿

## 任务入口

- 训练：`RobotLab-Isaac-Recover-Flat-TQBot2-v0`
- 回放：`RobotLab-Isaac-Recover-Flat-TQBot2-Play-v0`
- 算法：RSL-RL PPO，日志目录 `logs/rsl_rl/tqbot2_recover_flat/`
- 配置：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/recover/config/tqbot2/recover_env_cfg.py`

用户于 2026-09-18 明确：关节摆法无所谓，主要看机身朝向。当前目标是机身正面朝上、
贴近地面且稳定，不限制腿部关节角、关节速度、世界 yaw 或落点。
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

80% 回合在 SO(3) 上均匀采样姿态，覆盖侧卧、仰卧、头尾朝下及斜倾，
12 个关节在 URDF 限位内独立采样；20% 从接近趴姿的姿态开始，为学习提供成功样本。
所有回合随机 yaw、关节速度、根线速度和角速度；地面摩擦/恢复系数在启动时随机。
随机全姿态从 0.75 m 高度落地，近趴姿从 0.30 m 落地，避免初始地面穿透。
这包含落地过程，尚不是按真实倒地分布采样的静止地面状态数据集。
Play 将近趴姿采样比例设为零，并关闭策略观测噪声。

12 维动作按 FR/FL/RR/RL、hip/thigh/calf 的顺序控制。
动作先裁剪至 [-1, 1]，再映射为参考零点的关节偏移，最后按 URDF 限位裁剪位置目标。
hip/thigh/calf 缩放分别为 1.588/4.75/1.735 rad，可覆盖新版活动范围。
小腿参与控制。仿真 200 Hz，策略 50 Hz，单回合最长 12 s。

Recover 单独使用 hip/thigh PD 40/2、calf PD 60/3，保留源 URDF 对应的力矩/速度上限，
启用自碰撞，求解器迭代设为 8/2。原资产的公共 PD 默认值不受此覆盖影响。

Policy 观测共 42 维：投影重力 3、体坐标角速度 3、相对趴姿关节角 12、
关节速度 ×0.1 共 12、上一动作 12。Critic 额外接收基座线速度 3 和离地高度 1，共 46 维。
Policy 不需要外部定位或接触力输入。这个接口与源工程另一个 47 维 `RecoverPolicy` C++ 模块不同，
训练出的策略不能直接加载进该模块；部署时必须另行对齐观测、动作缩放、PD 和控制周期。

## 奖励与成功判据

奖励包含带正负方向的直立程度（权重 3）、低机身且机身低速的联合奖励（权重 6），
以及成功状态奖励（权重 4）和动作变化/力矩/限位惩罚。移除所有关节姿态匹配；
机身高度不超过 0.18 m 即不受高度惩罚，不再跟踪固定的 0.07 m。
使用重力 z 的符号，避免将仰卧误认为水平正立。所有身体触地都合法，不继承行走任务的触地终止。

必须同时满足以下条件并连续保持 0.5 s（25 控制步），才能以 `success` 结束回合：

- 投影重力 z < -0.95（倾角约小于 18.2°，yaw 自由）。
- trunk 离地高度处于 (0.02, 0.18) m，保留原任务的地面恢复要求。
- 根线速度 < 0.15 m/s，角速度 < 0.5 rad/s。
- 不检查关节角或关节速度；URDF 限位与力矩上限仍有效。
- 至少一个身体的地面净接触力 > 1 N。
- 回合已运行至少 1 s；空中短暂摆出姿态不算成功。

失败或重置会清空连续计数。恢复成功额外给一次 +30 奖励：Isaac 将配置权重乘以 dt，
所以 `success_bonus` 权重为 1500。这超过 gamma=0.99 下最高姿态维持奖励的折扣和 26，
避免策略为了继续获取逐步奖励而刻意拖延成功。越界、异常高度与 12 s 超时会结束回合。
TensorBoard 的 `Episode_Termination/success` 可用于查看训练回合成功比例；
训练混有容易初态，该比例不能代替纯随机初态的独立评估。

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
[0.03, 0.6]，初值 0.35；关闭额外熵奖励。PPO 的采样、log-prob、熵和 KL
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
