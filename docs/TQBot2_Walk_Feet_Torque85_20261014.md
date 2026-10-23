# TQBot2 行走独立 foot、训练力矩 85% 与当前训练入口

同日后续 StandardWalk 默认站姿已改为 thigh=0.80、calf≈-1.644557892 rad，默认足端 Z=-0.25 m，足高目标按 trunk 中默认位置加 8.5 cm 设为 -0.165 m；见 [站姿与足高参考帧](TQBot2_Walk_Stance_Height_20261014.md)。已运行实验的保存配置仍保持各自启动时的设置。

2026-10-14 按用户要求修改 `RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0`。本次仅准备和验证配置，未启动训练或实机部署。

## 四个 foot

原始 xacro/URDF 本来包含 `FR_foot`、`FL_foot`、`RR_foot`、`RL_foot`，每个 foot 为半径 0.012 m 的球体、质量 0.015 kg，固定于 calf 的 (0, 0, -0.153) m。此前 importer 合并固定关节，因此训练只有 13 个刚体，foot 球体与小腿、膝部碰撞形状共用 calf 接触统计。

新增 `TQBot2_walk.urdf`，只对四个 foot 固定关节增加 `dont_collapse=true`；质量、惯量、关节限位、视觉/碰撞几何不变。保留其他固定关节合并。生成器 `scripts/tools/build_tqbot2_walk_urdf.py` 和 `walk_manifest.json` 记录源与派生哈希；常规模型同步工具也会更新此派生模型。[NVIDIA UrdfJoint 文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/structisaacsim_1_1asset_1_1importer_1_1urdf_1_1_urdf_joint.html)说明固定关节可通过 dontCollapse 阻止合并。

StandardWalk 现为 17 个刚体、12 个驱动关节；足端奖励、步态配对、足端材质随机化全部选择 `.*_foot`。膝部/小腿接触进入非足端接触统计。自碰撞继续只允许左右前腿之间、左右后腿之间；拆出 foot 后允许配对从 18 对扩为 32 对，以保留这些腿间碰撞的含义。共 136 个刚体对，104 对禁用。通用资产与其他任务继续支持原 13 刚体模型。

位置图 `outputs/TQBot2_walk_feet_torque85_20261014/foot_positions.png` 直接根据当前 URDF 碰撞几何和正运动学生成，红色是 foot 球心，不是球面接触点，也不是膝盖。

## 训练参数

- 仅在 StandardWalk 环境配置中乘 `training_torque_scale=0.85`：髋/大腿 23.7 → 20.145 N·m，小腿 35.55 → 30.2175 N·m；同时用于 DC effort_limit、saturation_effort 和 PhysX effort_limit_sim。原 xacro、URDF 名义力矩、通用资产与硬件控制参数保持原值。
- 足端静摩擦 0.3–1.0、动摩擦 0.3–0.8；全身材料原有下限也是 0.3。继续不使用高摩擦平地。
- 明确设置 actor/critic 观测归一化 False，与 Go2 一致。归一化是用运行均值/方差变换网络输入，启用时部署必须带上同一组统计量；关闭时使用原有观测缩放。
- 当前 PPO 默认：learning_rate=0.001、schedule=adaptive、entropy_coef=0.01。StandardWalk 单独使用 `tqbot2_standard_walk` 日志目录、resume=False。

## 后续启动与部署的含义

之前所说的“冻结源码”是每次旧实验保存并运行的一份源码副本，不表示冻结神经网络、不进行学习。“启动参数”指当时执行 train.py 时传入的命令行覆盖值，旧组的固定 1e-4 和熵系数 0.005 不属于当前默认。

提供当前入口（在 isaaclab Python 环境中执行）：

```bash
bash scripts/reinforcement_learning/rsl_rl/train_tqbot2_walk.sh
```

该入口优先导入当前仓库源码，明确使用 0.001/adaptive/0.01，不引用旧快照。它默认创建新训练，不加载旧优化器或检查点；显式添加 resume/warm_start 参数属于另一次用户选择。学习率和熵系数用于训练更新，部署推理不使用它们；关闭观测归一化后，当前新策略不需要额外的运行均值/方差变换。训练力矩折扣不会被脚本写入硬件参数。

## 验证

- 6 项模型回归检查通过：派生 walk 模型仅多四个保留标志，其他物理 XML 一致，仍有 12 个驱动关节。
- 16 环境、300 步 PhysX 检查通过：四个 foot 独立存在，其位置与 calf + 固定偏移吻合；每足 15 g、总名义质量 13.3 kg；16 个克隆的 2176 对过滤关系正确；力矩限制、奖励、45→12 接口和部分复位有效。当前 PPO 配置与摩擦范围也已核对。
- 首次 PhysX 检查因测试预期张量和实际 default_mass 的 CPU/GPU 设备不同而失败，修正测试后通过；失败日志保留。模型生成及物理参数没有为该失败而改动。CPU 回归首次用系统 Python 时缺 scipy，改用已有 isaaclab 环境后通过，未安装依赖。
- 输出包含 `physics_check.json`、位置图及参数/源哈希记录。检查读取旧 model_14200 进行有限步集成验证，不证明新奖励下的训练或实机性能。
