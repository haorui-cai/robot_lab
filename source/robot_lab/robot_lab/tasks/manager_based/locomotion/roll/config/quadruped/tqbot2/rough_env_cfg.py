
import isaaclab.sim as sim_utils  # 物理材质、光照等仿真工具
import isaaclab.terrains as terrain_gen  # 地形生成器
from isaaclab.managers import RewardTermCfg as RewTerm  # 奖励项配置
from isaaclab.managers import SceneEntityCfg  # 场景实体选择器
from isaaclab.managers import TerminationTermCfg as DoneTerm  # 终止条件配置
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg  # 地形生成配置
from isaaclab.utils import configclass  # config 装饰器

from robot_lab.assets.tqbot import TQBOT2_CFG  # TQBot2 机器人配置
from robot_lab.tasks.manager_based.locomotion.roll.roll_env_cfg import RollRoughEnvCfg  # 父类环境配置

import robot_lab.tasks.manager_based.locomotion.roll.mdp as roll_mdp  # 翻滚专用MDP函数
from robot_lab.tasks.manager_based.locomotion.roll.mdp.base_acceleration import base_acceleration_rewards
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as vmdp  # 通用MDP函数（行走移植）

# 3cm凹凸地形（固定噪声，作为域随机化，不设难度课程）
NOISE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),  # 每个地块大小 8x8m
    border_width=20.0,  # 边界宽度
    num_rows=10,  # 10 行地块
    num_cols=20,  # 20 列地块
    horizontal_scale=0.1,  # 水平分辨率 0.1m
    vertical_scale=0.005,  # 垂直分辨率 0.005m
    slope_threshold=0.75,  # 坡度阈值
    use_cache=False,  # 不使用缓存
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(  # 随机均匀高度场
            proportion=1.0,  # 占全部地块的比例
            noise_range=(0.03, 0.03),  # 噪声幅度 ±3cm
            noise_step=0.01,  # 噪声步长
            downsampled_scale=0.5,  # 降采样尺度
            border_width=0.25,  # 子地形边界宽度
        ),
    },
)


@configclass
class TQBot2RollRoughEnvCfg(RollRoughEnvCfg):
    base_link_name = "trunk"  # 机身 body 名称
    foot_link_name = ".*_calf"  # 脚部 body 名称正则
    # fmt: off
    joint_names = [  # 全部 12 个关节，按顺序排列
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        super().__post_init__()  # 先执行父类初始化

        # 替换机器人模型为 TQBot2
        self.scene.robot = TQBOT2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 高度扫描仪的射线起点改为机身（原来默认是 base，但机器人没有 base link）
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/trunk"

        # ------------------------------Actions------------------------------
        # 不使用 URDF 默认关节位置作为 offset
        self.actions.joint_pos.use_default_offset = False
        # 综合仿真（平地速度/偏航、上下坡）的稳定连续滚动 command-q 统计：
        # 右 hip/thigh 固定于 (+1.57, +1.68)，左 hip 为 [-1.57, -0.71]、
        # 左 thigh 为 [1.68, 2.69]（均为 1--99%）。左右腿取并集后，让同类
        # 关节共享动作幅度；offset 仍分别置于各侧的运动中心，并保留约 15% 裕量。
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.50,
            ".*_thigh_joint": 0.60,
            ".*_calf_joint": 0.0,
        }
        self.actions.joint_pos.offset = {
            ".*R_hip_joint": 1.57,
            ".*L_hip_joint": -1.14,
            ".*R_thigh_joint": 1.68,
            ".*L_thigh_joint": 2.19,
            ".*_calf_joint": -2.82,
        }
        # 明确指定 12 个关节
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Observations------------------------------
        # 关节位置观测指定具体关节名
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # 关节速度观测指定具体关节名
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # 去掉地形高度扫描（平地翻滚不需要）
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # ------------------------------Commands------------------------------
        # 翻滚角速度 = 横向线速度 / 0.112 → ±0.784m/s 对应 ±7rad/s
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)  # 不要求前后移动
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.784, 0.784)  # 横向线速度指令范围
        self.commands.base_velocity.ranges.lin_vel_y = (0.70, 1.0)  # 横向线速度指令范围

        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)  # 偏航指令暂关掉
        self.commands.base_velocity.heading_command = False  # 不用航向指令模式
        self.commands.base_velocity.debug_vis = False  # 不显示指令可视化

        # ------------------------------Events------------------------------
        # 重置时关节位置在 0.9~1.0 倍之间随机
        self.events.randomize_reset_joints.params["position_range"] = (0.9, 1.0)
        # 重置时关节速度在 ±1.8 之间随机
        self.events.randomize_reset_joints.params["velocity_range"] = (-1.8, 1.8)
        # 重置时机身位姿和速度大范围随机
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
        }
        # 机身质量随机化
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # 其他 body 质量随机化
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [f"^(?!.*{self.base_link_name}).*"]
        # 机身质心随机化
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # 机身上下受到随机外力/力矩
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Terrain------------------------------
        self.scene.terrain.terrain_type = "plane"  # 使用平地（无边界，out_of_bounds 永不触发）
        # self.scene.terrain.terrain_type = "generator"  # (保留)噪声地形被注释掉了
        self.scene.terrain.terrain_generator = NOISE_TERRAIN_CFG  # 保留 terrain_generator 配置
        self.scene.terrain.max_init_terrain_level = None  # 不限制初始地形等级
        self.scene.terrain.collision_group = -1  # 所有环境在同一碰撞组
        # 地面物理材质：摩擦乘法混合，静/动摩擦系数各 1.0
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )

        # ------------------------------Rewards------------------------------
        # Use the corrected virtual base and fixed world axes for regularization.
        self.rewards.lin_vel_z_l2 = None
        self.rewards.roll_anti_jump = None
        self.rewards.base_height_above = None
        self.rewards.base_height_l2 = None
        self.rewards.body_lin_acc_l2 = None
        for name, term in base_acceleration_rewards().items():
            setattr(self.rewards, name, term)
        # 惩罚机身俯仰/偏航角速度（让翻滚更纯粹）
        # self.rewards.ang_vel_xy_l2.weight = 0
        # 不约束机身高度（翻滚时机身自然贴地）
        # self.rewards.base_height_l2 = None

        # 惩罚关节力矩过大
        self.rewards.joint_torques_l2.weight = -5.0e-4
        # 惩罚关节加速度过大（减少高频抖动）
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # 惩罚关节位置超限
        self.rewards.joint_pos_limits.weight = -5.0
        # 不奖励站立不动（翻滚任务不需要）
        # self.rewards.stand_still.weight = 0
        # 惩罚左右关节不对称，鼓励镜像运动
        self.rewards.joint_mirror.weight = -2.0
        self.rewards.joint_mirror.params["mirror_joints"] = [  # 镜像配对
            ["FL_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FR_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # 惩罚相邻帧动作变化太大（平滑控制信号）
        self.rewards.action_rate_l2.weight = -0.005

        # 不惩罚非期望接触（翻滚时需要全身触地）
        # self.rewards.undesired_contacts.weight = 0
        # 脚部接触力无关
        # self.rewards.contact_forces.weight = 0
        # 接触力传感器检测脚部
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 惩罚大腿空抖（高角速度 + 低触地力 = 无效抖动）
        self.rewards.roll_thigh_jitter = RewTerm(
            func=roll_mdp.roll_thigh_jitter, weight=-5e-3,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh"]), "contact_threshold": 25.0},
        )

        # 不奖励机身朝上（翻滚时不需要保持正立）
        self.rewards.upward.weight = 0

        # 禁用所有腿式行走相关奖励（翻滚用不到）
        # self.rewards.track_lin_vel_xy_exp = None  # 不追踪线速度
        # self.rewards.feet_air_time.weight = 0  # 不奖励腾空时间
        # self.rewards.feet_air_time_variance.weight = 0  # 不惩罚腾空时间差异
        # self.rewards.feet_gait.weight = 0  # 不奖励步态对称性
        # self.rewards.feet_contact.weight = 0  # 不奖励脚部触地规律
        # self.rewards.feet_contact_without_cmd.weight = 0  # 不惩罚无指令触地
        # self.rewards.feet_stumble.weight = 0  # 不检测绊倒
        # self.rewards.feet_slide.weight = 0  # 不惩罚脚滑
        # self.rewards.feet_height.weight = 0  # 不约束脚离地高度
        # self.rewards.feet_height_body.weight = 0  # 不约束脚相对机身高度

        # 追踪指令翻滚角速度 reward
        self.rewards.track_roll_vel_x = RewTerm(
            func=roll_mdp.track_roll_vel_x, weight=8.0,
            params={"command_name": "base_velocity", "std": 0.5, "roll_radius": 0.112},
        )

        # 存活奖励：3s~1, 5s~3, 10s~10
        self.rewards.survival_bonus = RewTerm(
            func=roll_mdp.survival_bonus, weight=2.0,
        )
        # 监控：当前存活时间（秒）
        self.rewards.survival_time_monitor = RewTerm(
            func=roll_mdp.survival_time_monitor, weight=1.0,
        )

        # 奖励高平均翻滚速度
        self.rewards.roll_vel_avg = RewTerm(
            func=roll_mdp.roll_vel_avg, weight=3.0,
        )

        # 惩罚平均滚速低于阈值
        self.rewards.slow_roll_penalty = RewTerm(
            func=roll_mdp.slow_roll_penalty, weight=2.0,
            params={"threshold": 1.0},
        )

        # 奖励持续朝指令方向翻滚超过 0.7s
        # self.rewards.roll_any = RewTerm(
        #     func=roll_mdp.roll_any, weight=1.5,
        #     params={"command_name": "base_velocity", "min_duration": 0.7},
        # )

        # Debug曲线在 TensorBoard 的 Debug/ 组下（由 roll_too_slow 写入 env.extras["log"]）

        # 跟踪指令 y 方向线速度（翻滚推进效率）
        self.rewards.track_lin_vel_y = RewTerm(
            func=roll_mdp.track_lin_vel_y, weight=8.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )


        # 偏航跟踪（暂禁用，后续通过课程学习加入）
        self.rewards.track_ang_vel_z_exp = None



        # 收腿姿态奖励（暂禁用）
        # self.rewards.roll_curled_posture = RewTerm(
        #     func=roll_mdp.roll_curled_posture, weight=1.0,
        # )

        # ------------------------------Terminations------------------------------
        # 走出地形边界时终止（平地模式下不会触发）
        self.terminations.terrain_out_of_bounds = DoneTerm(
            func=vmdp.terrain_out_of_bounds,
            params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
            time_out=True,
        )
        # 不用非法接触检测（翻滚全身都会接触地面）
        self.terminations.illegal_contact = None

        # 在滚动速度太慢中止条件
        self.terminations.roll_too_slow = DoneTerm(
            func=roll_mdp.roll_too_slow, time_out=False,
            params={"min_roll_vel_avg": 2.0},
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None  # 不用地形难度课程
        self.curriculum.command_levels = None  # 不用指令难度课程

        # ------------------------------Simulation------------------------------
        self.episode_length_s = 10.0  # 每个 episode 最长 10 秒

        # 如果当前类是 TQBot2RollRoughEnvCfg（而非子类），删掉所有 weight=0 的奖励项
        if self.__class__.__name__ == "TQBot2RollRoughEnvCfg":
            self.disable_zero_weight_rewards()
