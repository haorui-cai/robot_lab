"""Cycle rolling with four hips and two movable left thighs."""

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.tasks.manager_based.locomotion.roll.mdp import hip_only_cycle as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import joint_power

from .hip_only_cycle_env_cfg import CycleRewardsCfg, TQBot2RollHipOnlyCycleEnvCfg


@configclass
class LeftThighCycleRewardsCfg(CycleRewardsCfg):
    # The tracker gates yaw tracking by signed actual-cycle speed for this task.
    yaw_cycle_tracking = RewardTermCfg(func=mdp.cycle_yaw_reward, weight=5.0)
    joint_power = RewardTermCfg(func=joint_power, weight=-1.0e-4)


@configclass
class TQBot2RollLeftThighCycleEnvCfg(TQBot2RollHipOnlyCycleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.roll_range = (-14.0, -6.0)
        self.commands.base_velocity.actual_cycle_rewards = True
        self.commands.base_velocity.legacy_reward_fraction = 0.5
        self.commands.base_velocity.yaw_tracking_std = 0.125
        self.commands.base_velocity.yaw_success_tolerance = 0.06
        self.commands.base_velocity.yaw_reward_min_speed = min(
            abs(value) for value in self.commands.base_velocity.roll_range
        )
        self.commands.base_velocity.initial_speed_scales = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
        self.rewards = LeftThighCycleRewardsCfg()
        # Narrow physical randomization for this task and its slope variant.
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_com_positions.params["com_range"] = {
            axis: (-0.03, 0.03) for axis in ("x", "y", "z")
        }
        controlled = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint",
                      "FL_thigh_joint", "RL_thigh_joint"]
        self.joint_names = controlled
        self.scene.robot.spawn.asset_path = (
            f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/TQBot/TQBot2_description/urdf/TQBot2_left_thigh_roll.urdf"
        )
        self.scene.robot.init_state.joint_pos[".*L_thigh_joint"] = 1.68
        self.scene.robot.actuators["left_thigh"] = self.scene.robot.actuators["hip"].replace(
            joint_names_expr=[".*L_thigh_joint"]
        )
        self.actions.joint_pos.joint_names = controlled
        self.actions.joint_pos.scale = {".*_hip_joint": 1.0, ".*L_thigh_joint": 0.385}
        self.actions.joint_pos.offset = {
            ".*R_hip_joint": 0.52, ".*L_hip_joint": -0.52, ".*L_thigh_joint": 2.065,
        }
        self.actions.joint_pos.clip[".*L_thigh_joint"] = (1.68, 2.45)
        self.events.randomize_reset_base.params = {"height_table_name": "left_thigh_roll_spawn_heights.json"}
        self.events.randomize_reset_joints.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=controlled)}
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=controlled)
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["asset_cfg"].joint_names = controlled
            group.joint_vel.params["asset_cfg"].joint_names = controlled
            # Left shells now belong to separate movable thigh bodies.
            group.support_contacts.params = {"left_bodies": "(FL|RL)_(hip|thigh)"}
