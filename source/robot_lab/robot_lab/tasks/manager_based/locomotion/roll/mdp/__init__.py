"""MDP函数：给强化学习环境提供奖励函数和终止条件的计算逻辑"""

# Python 的 __future__ 语法，让类型注解中的类名可以写自己的名字（比如 ManagerBasedRLEnv）
from __future__ import annotations

# typing.TYPE_CHECKING：只在类型检查时导入，运行时跳过（避免循环导入）
from typing import TYPE_CHECKING

# torch = PyTorch，一个数学库，能在 GPU 上做矩阵运算
# torch.Tensor = 张量，类似于多维数组，所有数据都存成这种格式
import torch

# Articulation = 刚体关节链（四足机器人属于这类）
# RigidObject = 单个刚体
from isaaclab.assets import Articulation, RigidObject

# SceneEntityCfg = 一种配置，用来指定"选择哪个部位"（比如选哪几个关节、哪个body）
from isaaclab.managers import SceneEntityCfg

# ContactSensor = 接触传感器，能检测机器人和地面之间的接触力
from isaaclab.sensors import ContactSensor


# TYPE_CHECKING 为 True 时才执行下面的导入（写类型注解用，不会实际运行）
if TYPE_CHECKING:
    # ManagerBasedRLEnv = 强化学习环境的基类，env 参数就是它
    from isaaclab.envs import ManagerBasedRLEnv


def track_roll_vel_x(
    # env = 环境实例，包含机器人、传感器、物理引擎等所有信息
    env: ManagerBasedRLEnv,
    # std = 标准差，控制指数核的宽度，std 越大对误差的容忍度越大
    std: float,
    # command_name = 命令的名字，用来从命令管理器里读出指令值
    command_name: str,
    # roll_radius = 翻滚半径，把线速度指令换算成角速度目标的系数
    roll_radius: float,
    # asset_cfg = 指定读哪个机器人的数据，默认就是"robot"
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# -> torch.Tensor 表示这个函数的返回值类型是一个张量
) -> torch.Tensor:
    """
    奖励：实际翻滚角速度跟踪指令目标的精度
    返回值范围 [0, 1]，越高越好
    """
    # env.scene 是场景，里面放着所有实体（机器人、传感器等）
    # [asset_cfg.name] 按名字取出机器人对象（"robot"）
    # RigidObject 标注告诉 IDE 这个对象的类型
    asset: RigidObject = env.scene[asset_cfg.name]

    # .data 存着机器人每一时刻的状态数据
    # root_ang_vel_b = root angular velocity in body frame（机身坐标系下的角速度）
    # shape 是 (环境数, 3)，[:, 0] 取所有环境的 x 轴分量
    # x 轴 = 机身前后方向 = 翻滚轴
    roll_vel = asset.data.root_ang_vel_b[:, 0]

    # env.command_manager = 命令管理器，生成并存储指令值
    # .get_command(command_name) 返回形状 (环境数, 3) 的张量
    # 第 1 列（索引从 0 开始）是 lin_vel_y（横向线速度）
    # ÷ roll_radius（0.112m）把线速度转成角速度（ω = v / r）
    cmd_roll = env.command_manager.get_command(command_name)[:, 1] / roll_radius

    # torch.square = 每个元素求平方
    # roll_vel - cmd_roll = 实际值减去目标值 = 误差
    # 误差平方 = 正的误差值，越大说明跟踪得越差
    error = torch.square(roll_vel - cmd_roll)

    # torch.exp = e 的 x 次方（自然指数）
    # std ** 2 = std 的平方（Python 的 ** 就是乘方）
    # 公式：exp(-误差² / 标准差²)
    # 误差=0 时 → e⁰ = 1（满分）
    # 误差很大时 → e⁻ᵝᵍ → 接近 0
    return torch.exp(-error / std ** 2)


def survival_bonus(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """存活奖励，随时间平方增长：3s~1分, 5s~3分, 10s~10分"""
    step_dt = env.cfg.decimation * env.cfg.sim.dt
    t = env.episode_length_buf * step_dt
    return step_dt * t * 0.2  # 累计 ≈ 0.1 * t²


def survival_time_monitor(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """监控：返回当前存活时间（秒），TensorBoard值 = 当前env平均存活秒数"""
    step_dt = env.cfg.decimation * env.cfg.sim.dt
    return env.episode_length_buf * step_dt


def roll_vel_avg(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励：200步平均翻滚速度越大越好，除以10缩放到合理范围"""
    if not hasattr(env, "_vel_window"):
        return torch.zeros(env.num_envs, device=env.device)
    WINDOW = env._vel_window.shape[1]
    n_steps = torch.min(env.episode_length_buf + 1, torch.tensor(WINDOW, device=env.device))
    avg = torch.abs(env._vel_window.sum(dim=1) / n_steps.float())
    return avg / 5.0


def slow_roll_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚200步内平均滚速过低，低于threshold时给负奖励"""
    if not hasattr(env, "_vel_window"):
        return torch.zeros(env.num_envs, device=env.device)
    WINDOW = env._vel_window.shape[1]
    n_steps = torch.min(env.episode_length_buf + 1, torch.tensor(WINDOW, device=env.device))
    avg = torch.abs(env._vel_window.sum(dim=1) / n_steps.float())
    # 超过warmup(20步=0.4s)且低于阈值时惩罚
    gap = avg - threshold  # 负值 = 太慢
    return torch.where((env.episode_length_buf > 20) & (gap < 0), gap, torch.zeros_like(gap))


def base_height_above_l1(
    env: ManagerBasedRLEnv,
    # threshold = 高度阈值，超了就开始惩罚
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # height_scanner_base = 专门测机身离地高度的传感器
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_base"),
) -> torch.Tensor:
    """惩罚：机身高度超过阈值（防跳）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # root_pos_w[:, 2] = 世界系 z 坐标 = 机身离地高度
    base_height = asset.data.root_pos_w[:, 2]

    # torch.clamp(值, min=下限)：低于下限的就拉回到下限
    # 这里：高度 - 阈值，如果为负就 clamp 到 0
    # 结果：高度 <= 阈值 → 0（不惩罚）
    #       高度 > 阈值 → 超过的高度值
    return torch.clamp(base_height - threshold, min=0.0)


def roll_anti_jump(
    env: ManagerBasedRLEnv,
    vel_weight: float = 2.0,
    acc_weight: float = 5.0e-3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚：机身 z 轴速度和加速度（防跳）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # root_lin_vel_b[:, 2] = 机身系 z 轴速度（上下跳动）
    lin_vel_z = asset.data.root_lin_vel_b[:, 2]

    # root_lin_acc_b 可能不存在，所以用 hasattr 判断
    # 存在就读取 z 轴加速度，不存在就创建一个全 0 张量
    # torch.zeros_like(lin_vel_z)：创建一个形状和 lin_vel_z 一样的全 0 张量
    lin_acc_z = asset.data.root_lin_acc_b[:, 2] if hasattr(asset.data, "root_lin_acc_b") else torch.zeros_like(lin_vel_z)

    # vel_weight × 速度² + acc_weight × 加速度²
    # 速度和加速度越大惩罚越大，两个系数控制各自的比重
    return vel_weight * (lin_vel_z**2) + acc_weight * (lin_acc_z**2)




def roll_hip_asymmetry(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_thigh"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励：大腿发力时左右髋关节位置差（当前未使用）"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_thigh_force = torch.max(torch.norm(net_forces, dim=-1).max(dim=1).values, dim=-1).values
    # tanh(力/10)：力大时接近1，力小时接近0，作为门控系数
    contact_gate = torch.tanh(max_thigh_force / 10.0)

    # Articulation 和 RigidObject 的区别：Articulation 有关节（joint），可以查关节索引
    asset: Articulation = env.scene[asset_cfg.name]

    # .find_joints("正则表达式") 返回两个值：(关节索引列表, 关节名列表)
    # [0] 取索引列表，用来从 joint_pos 张量中取对应关节的位置
    # ".*L_hip_joint" 匹配 FL_hip_joint 和 RL_hip_joint（左腿髋关节）
    l_hip_ids = asset.find_joints(".*L_hip_joint")[0]
    # ".*R_hip_joint" 匹配 FR_hip_joint 和 RR_hip_joint（右腿髋关节）
    r_hip_ids = asset.find_joints(".*R_hip_joint")[0]

    # joint_pos[:, 左hip索引] = 取所有环境的左髋关节位置
    # torch.mean(..., dim=1) = 在关节维度求平均（两个左髋→平均成一个数）
    # dim=1 表示在第二个维度（关节维度）上求均值
    left_hip = torch.mean(asset.data.joint_pos[:, l_hip_ids], dim=1)
    right_hip = torch.mean(asset.data.joint_pos[:, r_hip_ids], dim=1)
    # 左右髋关节位置的绝对差值
    hip_diff = torch.abs(left_hip - right_hip)

    # 门控系数 × 髋差异 = 只有大腿在发力时才奖励髋不对称
    return contact_gate * hip_diff


def roll_thigh_drive(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_thigh"),
    max_duration: float = 0.1,
) -> torch.Tensor:
    """奖励：大腿蹬地，每次接触只奖励前0.1秒（当前未使用）"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_thigh_force = torch.max(torch.norm(net_forces, dim=-1).max(dim=1).values, dim=-1).values

    # hasattr(对象, 属性名) = 检查对象是否有这个属性
    # 这里用来检查 env 上有没有计时器（有的话说明上一步已经创建了）
    if not hasattr(env, "roll_thigh_contact_timer"):
        # torch.zeros(环境数, device=env.device) 创建全0张量
        # env.device 是计算设备（"cuda:0" 或 "cpu"）
        # 创建的数据要跟环境在同一个设备上
        env.roll_thigh_contact_timer = torch.zeros(env.num_envs, device=env.device)

    # 接触力 > 25N 才算有效蹬地
    in_contact = max_thigh_force > 25.0

    # env.cfg.decimation = 每个动作之间跑几步物理仿真（当前是4）
    # env.cfg.sim.dt = 物理仿真的步长时间（当前0.005s）
    # decimation × dt = 每步的真实时间 = 0.02s
    step_dt = env.cfg.decimation * env.cfg.sim.dt

    # torch.where(条件, 真时取值, 假时取值)
    # 类似 C 语言的三目运算符 condition ? true_val : false_val
    # 这里：正在接触→计时器+dt，没接触→计时器归零
    # torch.zeros_like(env.roll_thigh_contact_timer) 创建一个和传入张量形状一样的全0张量
    env.roll_thigh_contact_timer = torch.where(
        in_contact,
        env.roll_thigh_contact_timer + step_dt,
        torch.zeros_like(env.roll_thigh_contact_timer),
    )

    # 当前接触时间是否还在奖励窗口内（≤0.1s）
    within_window = env.roll_thigh_contact_timer <= max_duration
    # torch.where：窗口内就给 tanh(力/50)，窗口外给 0
    reward = torch.where(within_window, torch.tanh(max_thigh_force / 50.0), torch.zeros_like(max_thigh_force))
    return reward


def roll_curled_posture(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励：膝关节收拢姿态（当前未使用）"""
    asset: Articulation = env.scene[asset_cfg.name]
    # joint_pos = 所有关节的位置，shape (环境数, 关节数)
    joint_pos = asset.data.joint_pos

    # 找到各类关节的索引
    r_hip_ids = asset.find_joints(".*R_hip_joint")[0]    # 右髋
    l_hip_ids = asset.find_joints(".*L_hip_joint")[0]    # 左髋
    thigh_ids = asset.find_joints(".*_thigh_joint")[0]    # 大腿（所有腿）

    # 初始化全0奖励
    reward = torch.zeros(env.num_envs, device=env.device)

    # torch.exp(-(位置 - 目标)²)：位置越接近目标值→exp(≈0)→接近1
    # .mean(dim=1)：如果有多个同类型关节，在关节维度上求平均
    # +=：累加到总奖励上
    reward += torch.exp(-torch.square(joint_pos[:, r_hip_ids] - 1.58)).mean(dim=1)
    reward += torch.exp(-torch.square(joint_pos[:, l_hip_ids] + 1.58)).mean(dim=1)
    reward += torch.exp(-torch.square(joint_pos[:, thigh_ids] - 1.67)).mean(dim=1)

    # 除以 3 取平均（3 组关节的平均分）
    reward /= 3.0
    return reward


def roll_thigh_jitter(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 25.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚大腿无效抖动：高角速度 + 低触地力 = 空抖，低角速度 + 高触地力 = 有效蹬地。"""
    # 大腿触地力
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_contact = torch.norm(contact_forces, dim=-1).max(dim=1).values  # (num_envs, num_thighs)
    contact_weight = torch.tanh(max_contact / contact_threshold)  # 0~1, 力越大越接近 1

    # 大腿关节速度
    asset: Articulation = env.scene[asset_cfg.name]
    thigh_ids = asset.find_joints(".*_thigh_joint")[0]
    thigh_vel = asset.data.joint_vel[:, thigh_ids]  # (num_envs, 4)

    # 惩罚 = 大腿速度 × (1 - 触地权重)
    #   触地蹬地时 → contact_weight≈1 → (1-weight)≈0 → 不惩罚
    #   空抖时 → contact_weight≈0 → (1-weight)≈1 → 按速度惩罚
    jitter = torch.abs(thigh_vel) * (1.0 - contact_weight)
    return jitter.mean(dim=1)


def roll_yaw_rate(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """偏航角速度跟踪奖励（当前禁用，后续通过课程学习加入）"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # root_ang_vel_w[:, 2] = 世界系 z 轴角速度（真正的偏航率，翻滚时也正确）
    # 相比 root_ang_vel_b[:, 2]（机身系z轴）不会随翻滚而改变
    error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-error / std ** 2)


def track_lin_vel_y(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励：跟踪指令 y 方向线速度（衡量翻滚推进效率）。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # 机身系 y 轴速度 = 横向滑行速度（翻滚的直接效果）
    lin_vel_y = asset.data.root_lin_vel_b[:, 1]
    # 指令也是 y 方向线速度
    cmd_y = env.command_manager.get_command(command_name)[:, 1]
    # 指数奖励：误差越小越接近 1
    error = torch.square(lin_vel_y - cmd_y)
    return torch.exp(-error / std ** 2)


def roll_assist_torque(
    env: ManagerBasedRLEnv,
    command_name: str,
    # max_torque = 最大辅助力矩（Nm），训练开始时施加
    max_torque: float = 2.0,
    # decay_steps = 衰减步数，超过后力矩降为 0
    decay_steps: float = 100000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """辅助翻滚力矩 + 方向匹配奖励，力矩随训练步数衰减。

    训练早期给机身 x 轴施加恒定力矩，辅助机器人朝指令方向翻滚。
    随训练进行力矩逐渐减小到 0，迫使策略学会自主翻滚。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # 指令 y 方向线速度决定翻滚方向
    cmd_y = env.command_manager.get_command(command_name)[:, 1]
    # 当前翻滚角速度（机身 x 轴）
    roll_vel = asset.data.root_ang_vel_b[:, 0]

    # 衰减系数：common_step_counter 从 0 → decay_steps 线性衰减到 0
    progress = min(env.common_step_counter / decay_steps, 1.0)
    decay = 1.0 - progress  # 1.0 (开始) → 0.0 (训练结束)

    # 施加辅助力矩（decay=0 时力矩自动为 0）
    body_ids = asset_cfg.body_ids  # SceneEntityCfg 解析后的 body 索引
    torque = torch.zeros(env.num_envs, len(body_ids), 3, device=env.device)
    torque[:, :, 0] = (decay * max_torque * torch.sign(cmd_y)).unsqueeze(-1)
    asset.set_external_force_and_torque(
        forces=torch.zeros_like(torque),
        torques=torque,
        body_ids=body_ids,
    )

    # 奖励：翻滚方向与指令一致时给分，也随训练衰减
    alignment = torch.clamp(roll_vel * cmd_y, min=0.0)
    reward = decay * torch.tanh(alignment / 0.5)
    return reward


def roll_any(
    env: ManagerBasedRLEnv,
    command_name: str,
    # min_duration = 持续滚动多久后才开始给分（秒）
    min_duration: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励：朝指令方向持续滚动超过一定时间后给分"""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll_vel = asset.data.root_ang_vel_b[:, 0]              # 当前翻滚角速度

    # get_command 返回 (环境数, 3)，[:, 1] 取 lin_vel_y 列
    cmd = env.command_manager.get_command(command_name)[:, 1]

    # roll_vel × cmd：同号时为正（方向一致），异号为负（方向相反）
    # >= 0.01：留一个小容差，避免指令接近0时误判
    # 结果是 bool 张量，True=方向对了，False=方向错了
    same_sign = (roll_vel * cmd) >= 0.01

    # 创建计时器（每个环境独立计时）
    if not hasattr(env, "roll_any_timer"):
        env.roll_any_timer = torch.zeros(env.num_envs, device=env.device)

    # decimation = 4（每个动作步包含4个物理子步）
    # sim.dt = 0.005s（每个物理子步的时间）
    # step_dt = 4 × 0.005 = 0.02s（每个动作步的真实时间）
    step_dt = env.cfg.decimation * env.cfg.sim.dt

    # torch.where：方向对了就让计时器 + dt（累计），方向错了就归零
    env.roll_any_timer = torch.where(
        same_sign,
        env.roll_any_timer + step_dt,
        torch.zeros_like(env.roll_any_timer),
    )

    # 是否已经持续滚够了 min_duration 秒
    # >= 比较的结果也是 bool 张量
    sustained = env.roll_any_timer >= min_duration

    # torch.where：持续够了就给 tanh(角速度/1)，没持续够就给 0
    # torch.abs = 绝对值（不管快慢，只看大小）
    reward = torch.where(sustained, torch.tanh(torch.abs(roll_vel) / 1.0), torch.zeros_like(roll_vel))
    return reward


def roll_too_slow(
    env: ManagerBasedRLEnv,
    min_roll_vel_avg: float,
    warmup: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    失败检测：在第3/6/9秒检查净平均翻滚速度
    净平均速度 = |累计净角度| ÷ 时间
    如果低于阈值则终止这轮，让机器人重新开始
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # root_ang_vel_b[:, 0] = 机身x轴角速度，即翻滚速度（rad/s）
    roll_vel = asset.data.root_ang_vel_b[:, 0]

    # cfg = 环境配置
    # decimation = 4（每个动作步包含的物理子步数）
    # sim.dt = 0.005（物理仿真步长，秒）
    # 两者相乘 = 0.02s = 每个动作步的真实时间
    step_dt = env.cfg.decimation * env.cfg.sim.dt

    # episode_length_buf = 当前 episode 已经跑了多少步
    # × step_dt = 当前 episode 已经跑了多少秒
    current_time = env.episode_length_buf * step_dt

    # --- 滑动窗口：最近200步的 roll_vel ---
    WINDOW = 200
    if not hasattr(env, "_vel_window"):
        env._vel_window = torch.zeros(env.num_envs, WINDOW, device=env.device)
        env._vel_window_idx = 0

    reset_mask = env.episode_length_buf == 0
    # reset的环境清零整个窗口
    env._vel_window[reset_mask] = 0.0

    # 写入当前位置
    env._vel_window[:, env._vel_window_idx] = roll_vel
    env._vel_window_idx = (env._vel_window_idx + 1) % WINDOW

    # 前200步用实际步数做分母
    n_steps = torch.min(env.episode_length_buf + 1, torch.tensor(WINDOW, device=env.device))
    roll_vel_avg = torch.abs(env._vel_window.sum(dim=1) / n_steps.float())

    # --- TensorBoard Debug组 ---
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Debug/roll_vel_x"] = roll_vel.mean().item()
    env.extras["log"]["Debug/roll_vel_avg"] = roll_vel_avg.mean().item()

    # --- 持续检测：超过warmup秒后，只要100步平均速度低于阈值就终止 ---
    result = (current_time > warmup) & (roll_vel_avg < min_roll_vel_avg)
    return result
