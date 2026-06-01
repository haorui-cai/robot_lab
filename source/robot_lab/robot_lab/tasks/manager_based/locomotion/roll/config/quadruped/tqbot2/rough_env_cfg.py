
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from robot_lab.assets.tqbot import TQBOT2_CFG
from robot_lab.tasks.manager_based.locomotion.roll.roll_env_cfg import RollRoughEnvCfg

import robot_lab.tasks.manager_based.locomotion.roll.mdp as roll_mdp
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as vmdp

# Fixed 3cm fractal noise terrain (acts as domain randomization, no difficulty curriculum)
NOISE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.03, 0.03),
            noise_step=0.01,
            downsampled_scale=0.5,
            border_width=0.25,
        ),
    },
)


@configclass
class TQBot2RollRoughEnvCfg(RollRoughEnvCfg):
    base_link_name = "trunk"
    foot_link_name = ".*_calf"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = TQBOT2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/trunk"

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = 1.0
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # Remove height_scan (not needed for flat rolling)
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # ------------------------------Commands------------------------------
        # roll rate = lin_vel_y / roll_radius → ±7 rad/s corresponds to ±0.784 m/s
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.784, 0.784)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        # ------------------------------Events------------------------------
        self.events.randomize_reset_joints.params["position_range"] = (0.9, 1.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (-1.8, 1.8)
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [f"^(?!.*{self.base_link_name}).*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Terrain------------------------------
        self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_type = "generator"  # disabled: noise terrain
        self.scene.terrain.terrain_generator = NOISE_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.collision_group = -1
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )

        # ------------------------------Rewards------------------------------
        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.base_height_l2 = None

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -5.0e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.stand_still.weight = 0
        self.rewards.joint_mirror.weight = -0.8
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FR_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact
        self.rewards.undesired_contacts.weight = 0
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0

        # Disable velocity-tracking and feet rewards (not needed for rolling)
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time_variance.weight = 0
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = 0

        # Roll-specific rewards (command tracking)
        # roll rate target = lin_vel_y / 0.112
        self.rewards.roll_rate = RewTerm(
            func=roll_mdp.roll_rate, weight=3.0,
            params={"command_name": "base_velocity", "std": 0.5, "roll_radius": 0.112},
        )
        self.rewards.roll_attitude = None

        # Yaw tracking
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=vmdp.track_ang_vel_z_exp, weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )

        # Anti-jump
        self.rewards.base_height_above = RewTerm(
            func=roll_mdp.base_height_above_l1, weight=-2.0,
            params={"threshold": 0.18, "asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("height_scanner_base")},
        )
        self.rewards.roll_anti_jump = RewTerm(
            func=roll_mdp.roll_anti_jump, weight=-1.0,
            params={"vel_weight": 2.0, "acc_weight": 5.0e-3},
        )

        # Contact bonus for body+legs during rolling
        self.rewards.roll_contact_bonus = RewTerm(
            func=roll_mdp.roll_contact_bonus, weight=0.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", "trunk"])},
        )

        # ------------------------------Terminations------------------------------
        self.terminations.terrain_out_of_bounds = DoneTerm(
            func=vmdp.terrain_out_of_bounds,
            params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
            time_out=True,
        )
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        # ------------------------------Simulation------------------------------
        self.episode_length_s = 10.0

        if self.__class__.__name__ == "TQBot2RollRoughEnvCfg":
            self.disable_zero_weight_rewards()
