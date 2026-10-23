# TQBot2 自由步态对照：另一台电脑训练说明

本分支准备两个**随机初始化**任务。`free` 只从已经核准的 V4 配置删除步态塑形；`v4` 保留 V4 作为配对基线。没有 HIM、模仿学习或周期命令输入。本机只运行原 V4 从头实验，**没有启动这里的 free 训练**。自由探索不保证自动得到低步频或好步态，需要按速度、稳定性、滑移和坡面表现验收。

## 1. 版本与私有模型

公开代码仓库：`https://github.com/haorui-cai/robot_lab`；分支：`codex/tqbot2-free-gait-20261023`。该分支基于已发布的 `a33986edd674b64e69ca7b36bc5344dc3d11680a`。首次取回后记录 `git rev-parse HEAD`，后续复现检出该完整 SHA，不依赖可能移动的分支名称。新增的 `reward_v4_env_cfg.py` 是冻结 V4 原配置的独立副本，只改类名；恢复原类名后的 SHA256 为 `0295159f9dcbbb70c8fd59b8f24985c2067ce93672d988b07037c766d24c57c7`，CPU 测试检查它。

固定依赖：Ubuntu22.04 / Python3.11，IsaacSim5.1.0.0，IsaacLab commit `84d0ff05a6511d52a31090ce586686ceba56cd84`（包版本0.54.3），RSL-RL5.0.1，PyTorch2.7.0 / CUDA12.8。主要包约束在 [TQBot2_training_constraints.txt](TQBot2_training_constraints.txt)。这是当前运行环境的关键版本约束，**不是已在干净电脑验证的完整环境锁文件**；安装后保存 `pip freeze`、驱动版本与 `git status --short` 到自己的实验记录。

机器人依赖是既有私有子模块 `ustc-407lab/tqbot_description`，固定 commit `7843022e8c9b58cd1fb4373090366022302a7cd6`。本分支不包含其 URDF、Xacro、网格或生成资产。先通过组织正常权限流程获得此私有仓库的读取权限，并配置自己已有的 GitHub SSH 访问；不要将密钥、token 或私有模型提交到公开仓库。无需创建新凭据。其他私有模型副本不能直接替代此 pin。

```bash
git clone --branch codex/tqbot2-free-gait-20261023 --single-branch https://github.com/haorui-cai/robot_lab.git
cd robot_lab
git rev-parse HEAD
git submodule update --init external/tqbot_description
git -C external/tqbot_description rev-parse HEAD
```

模型生成入口是 `python scripts/tools/sync_tqbot2_description.py`。它验证子模块 commit 与源文件无修改，并调用其构建器。版本化符号链接 `source/robot_lab/data/Robots/TQBot/TQBot2_description` 指向子模块的忽略目录 `build/isaac/TQBot2_description`。这是 **TQBot2**，12关节；脚名 `FL_foot/FR_foot/RL_foot/RR_foot`，每腿 `hip_joint/thigh_joint/calf_joint`，配置保留自己的顺序映射。不要用 Go2 或 TQBot1 资产替换。详情见 [模型依赖说明](../TQBot1_TQBot2_Description_Dependency.md)。

## 2. 创建环境

以下命令在另一台电脑执行；本次没有安装或执行新的第三方代码。先按固定 IsaacLab checkout 自带的安装文档检查操作系统、驱动和显卡支持，并接受 IsaacSim 正常许可流程。此示例假设使用 conda 和 pip 版 IsaacSim：

```bash
conda create -n tqbot2-free python=3.11 -y
conda activate tqbot2-free
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0.0' --extra-index-url https://pypi.nvidia.com
# 在 robot_lab 的同级目录安装 IsaacLab；不要放进 robot_lab 内。
cd ..
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 84d0ff05a6511d52a31090ce586686ceba56cd84
./isaaclab.sh --install rsl_rl
cd ../robot_lab
python -m pip install -c docs/experiments/TQBot2_training_constraints.txt rsl-rl-lib==5.0.1 xacrodoc==2.0.1 toml==0.10.2 PyYAML==6.0.2 tensorboard==2.20.0
# 仅安装本项目本身；避免 setup.py 的无关 AMP/CusRL 依赖改变本实验环境。
python -m pip install --no-deps -e source/robot_lab
python scripts/tools/sync_tqbot2_description.py
python -m unittest discover -s scripts/tests -p 'test_tqbot2_*control.py' -v
python -m unittest discover -s scripts/tests -p 'test_tqbot2_cadence_audit.py' -v
nvidia-smi
```

如果解析器报告与约束冲突，停止并保留报错，不默默升级 PPO、IsaacLab 或 Torch。CPU 测试不需要私有模型，但训练必须完成模型构建。上述干净安装和新 recorder 的 GPU 集成尚未验证，首次只做小规模 smoke 后再扩大。

## 3. 与 V4 的确切差异

冻结 V4 的21项活动奖励中删除以下7项，其余14项数值和参数保留。机器可读原始合同见 [TQBot2_V4_reward_contract.json](TQBot2_V4_reward_contract.json)。`free_gait_policy.py` 遇到未经审查的新增活动奖励会直接报错。

| 删除项 | V4权重 | 删除原因 |
|---|---:|---|
| feet_frequency | +0.2 | 显式2Hz完整周期目标 |
| feet_gait | +0.5 | 对角脚同步、两组反相 |
| feet_air_time_variance | −1 | 约束各脚腾空/接触时长一致性 |
| joint_mirror | −0.05 | 实际是对角脚位移对称，不是普通限位 |
| joint_pos_penalty | −0.2 | 实际为 SwingJointPositionPenalty，按接触相位豁免摆腿 |
| feet_ground_clearance | −0.5 | 5cm摆脚目标和摆动相位门控，属于步态风格塑形 |
| feet_contact_without_cmd | +0.1 | 实际奖励首次触地事件，并非持续支撑；避免零命令下触地点数激励 |

保留：`lin_vel_z_l2`, `ang_vel_xy_l2`, `joint_torques_l2`, `joint_acc_l2`, `joint_pos_limits`, `joint_power`, `stand_still`, `action_rate_l2`, `undesired_contacts`, `contact_forces`, `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `feet_slide`, `upward`。

边界：`feet_slide` 用当前接触状态惩罚支撑脚相对静止地面的滑移，没有目标频率、占空比或脚对时序；`contact_forces` 是瞬时载荷/冲击成本；`stand_still` 只在低命令下约束关节姿态。这些通用约束仍可能间接影响策略选择的步态，但没有规定节奏。已有未启用的 air-time/contact/height/frequency-excess/cycle-overdue 等塑形也保持关闭。

机器人、碰撞、关节/力矩限制、动作、观测、命令、地形课程、随机化、PPO均继承冻结 V4。物理dt0.005s、decimation4、控制50Hz；actor45维、critic235维，MLP512/256/128，身份归一化；标称PD20/0.5并保留 V4 的增益随机化，不能与旧部署PD14/0.6混用。原 V4 的终止项为 time_out 与 terrain_out_of_bounds，`illegal_contact` 原本就是关闭的，本实验没有新增或关闭终止保护。身体非期望接触惩罚、物理限制和原有碰撞保持。地形包含原有台阶、粗糙面及坡面课程，坡面上限30度是训练配置上限，**不是已验证可爬30度**。

## 4. 从头启动与显存

命令默认只打印计划；必须显式 `--execute` 才启动。以下 free 命令只用于另一台电脑：

```bash
python scripts/TQBot2_train_control.py --variant free --num-envs 4096 --iterations 12000 --seed 42 --run-name free_seed42
# 检查上面的计划后执行；建议首次先用128环境/2次更新做集成smoke。
python scripts/TQBot2_train_control.py --variant free --num-envs 128 --iterations 2 --seed 42 --run-name free_smoke --execute
python scripts/TQBot2_train_control.py --variant free --num-envs 4096 --iterations 12000 --seed 42 --run-name free_seed42 --execute
# 可选的同机配对基线，须在资源允许时单独运行：
python scripts/TQBot2_train_control.py --variant v4 --num-envs 4096 --iterations 12000 --seed 42 --run-name v4_seed42 --execute
```

4096环境/24步PPO rollout、12000更新与原 V4预算一致。16GB显卡是本机参考，其他显卡未测量；脚本暂以4096环境需8GiB空闲、2048需6GiB、其余需4GiB为保守启动门槛，**不是显存用量保证**。显存不足显式改 `--num-envs 1024` 或更少，并在记录中注明批量变化；不停止别人的进程，不自动重试或倍增seed。原 V4 本机早期每次更新约5秒、全程约17小时，另一电脑时间未知。启动前检查 `nvidia-smi`，不要把驱动状态与容器设备可见性混为一谈。

`train_tqbot2_ablation.py` 在任何 checkpoint load 前审计：resume=false、warm_start为空、load_run=`__NO_RESUME__`、checkpoint=`__NO_CHECKPOINT__`、迭代0、Adam状态空、身份归一化、奖励集合正确。将 `fresh_initialization.json` 和 `initial_random_state.pt` 写到新run；保存actor/critic指纹、资产hash、代码commit、seed、环境数。审计失败就停止，不回退加载旧权重。每100次更新写checkpoint，不覆盖旧run。smoke与正式训练是两个独立随机初始化run。

```bash
# 训练记录；将 RUN 替换为实际新目录，不要使用latest通配选择旧实验。
RUN=logs/rsl_rl/tqbot2_free_scratch/实际新run目录
cat "$RUN/fresh_initialization.json"
python -m tensorboard.main --logdir logs/rsl_rl/tqbot2_free_scratch
```

恢复只用于**本次 free run 意外中断**，与从头实验分开记录。使用普通train入口，明确同一run和checkpoint；不要通过fresh-only入口恢复，不加载V3/V4或HIM。RSL-RL的 `max_iterations` 为本次追加更新数，先确认已完成数，再填剩余预算：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Rough-TQBot2-FreeGaitScratch-v0 --headless --device cuda:0 --num_envs 4096 --seed 42 --experiment_name tqbot2_free_scratch --resume --load_run '精确旧run目录名' --checkpoint 'model_100.pt' --max_iterations 11899 agent.check_for_nan=true env.commands.base_velocity.debug_vis=false
```

恢复脚本创建新的日志目录；记录父run、checkpoint SHA256和追加预算。保留旧目录、配置及初始审计，不用宽泛的 `.*` 自动挑选策略。

## 5. 回放与可靠步频/坡面评估

普通有限步数回放（加载可信的自有checkpoint）：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-TQBot2-FreeGaitScratch-v0 --checkpoint "$RUN/model_100.pt" --num_envs 8 --max_steps 400 --headless
```

附带的**新可移植 recorder 仅做过CPU语法检查，尚未在GPU运行**。先验证一次小规模录制的样本数/脚顺序/时间戳，再用于量化；它从已有200Hz录制方案移植，单独进行有界仿真，不接触实机：

```bash
python scripts/reinforcement_learning/rsl_rl/record_tqbot2_ablation.py --variant free --checkpoint "$RUN/model_100.pt" --output eval/free100_speed05 --speed .5 --seconds 8 --num_envs 24 --headless --device cuda:0
python scripts/tools/analyze_tqbot2_cadence.py eval/free100_speed05/trace.npz --output eval/free100_speed05/cadence.json
# 坡面单独测试；名义角度不等于机器人实际到达斜坡。
python scripts/reinforcement_learning/rsl_rl/record_tqbot2_ablation.py --variant free --checkpoint "$RUN/model_100.pt" --output eval/free100_slope05 --terrain slope --slope-deg 5 --speed .3 --seconds 12 --num_envs 8 --headless --device cuda:0
```

录制协议：1秒零命令稳定，然后固定命令；停用观测噪声、外推和自动reset，保留固定seed下的质量/COM/材质/PD随机化，记录实际失败样本。**这些仅为评估改动，不进入训练配置**。检查 `evaluation_env.yaml` 与 `metadata.json`，同一seed/环境/地形比较配对策略。nominal slope 的金字塔有平顶，必须检查 `local_slope_deg`、轨迹和位移，不能把停在平顶算爬坡成功。

分析输入为不含pickle的NPZ：`t[T]`, `foot_force[T,N,4]`（N）, `foot_z_body[T,N,4]`（m）, `velocity_b[T,N,3]`, `upright[T,N]`（−gravity_z）；可选 `episode[T,N]` 用于隔断reset、`command[T,N,3]`、`foot_tangent_speed[T,N,4]`。recorder写全部必要字段，物理采样200Hz，不拿控制50Hz当步频。

分析器丢弃前2秒，以实际水平速度0.05–0.25、0.25–0.5、≥0.5m/s分组，只接受完整且连续的稳态周期。触地到下次同脚触地才算一个周期；分别报告接触周期与“至少40ms腾空/支撑、身体坐标脚高度摆幅≥1cm”的运动学检查周期。缺失/静止/跌倒不能记作0Hz成功。运动学检查仅是诊断，不等于自然好步态证明；坡面上再结合局部净空、法向与视频。

同时报告1/.5N、3/1N、5/2N滞回阈值各20/40ms确认的6种结果、周期CV、占空比、样本数、失败比例及过滤敏感性。若频率范围差异>10%标记 filter_sensitive（诊断阈值，并非用户验收指标）；不挑最低值作真值。策略之间需比较相同速度、共同有效环境，另外报告全部失败/无有效周期数量，不能只比较剩余成功样本。

建议检查点100/1000/3000/6000/11999；先确认站稳和速度，再看步频下降是否以丢失速度、滑移或小碎步为代价。建议速度0、0.15、0.3、0.5m/s，固定命令稳定段与加减速段分开；每脚查看周期分布、抬脚净空、接触力、滑移，连同速度RMSE、姿态/跌倒、关节力矩与动作突变。0速度主要评估安静站立，不比较行走Hz。

建议坡度0/5/10/15/20度逐级增加，必要时再尝试更高；这些是提出的测试阶梯，非用户指定能力承诺。上坡、下坡、横坡分别验证（recorder当前固定初始朝向，后两者需扩展/核实方向设置后使用）。每组至少10个固定seed/起点试次，记录接触到真实斜面的时间、沿坡有效前进距离、完成率、滑移、速度误差、roll/pitch、跌倒和越界；成功必须到达真实斜面并通过规定距离，不能只“未触发termination”。优先用相同资产/接触/PD下V4配对验证，再考虑真实硬件；此分支不部署或启动实机。

## 6. 已验证的范围

- CPU：8项配置/初始化拒绝/命令测试、7项完整周期/去抖/缺失/重置边界测试；V4副本hash、7项删除/14项保留、默认计划不启动训练；所有新增Python语法检查。
- CPU测试使用奖励合同fixture与纯Python操作，**没有宣称实例化了完整Isaac环境或实际跑过free PPO update**。
- 旧实测200Hz轨迹可供离线分析器回归；新 recorder 的GPU集成、干净机器安装、free训练、坡面成功率均未验证。
- 没有权重、日志、私有资产或认证信息被提交。现有V4训练、旧配置/部署指针不受此分支影响。
