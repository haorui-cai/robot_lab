"""Standard velocity-tracking walking task for TQBot2.

This task deliberately uses the repository's regular joint-position action
term.  Recovery and rolling remain separate skills so that their specialised
action mappings do not alter the walking policy's MDP.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from robot_lab.assets.tqbot import TQBOT2_CFG
from robot_lab.assets.tqbot_collision import spawn_tqbot2_walk_with_leg_collisions
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.foot_ground import FootGroundMotion
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.high_friction_flat import (
    HighFrictionFlatMaterial, high_friction_flat_exit,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.gentle_recovery import (
    RecoveryTorqueChange, recovery_joint_effort, recovery_joint_velocity,
    walking_energy_efficiency, walking_joule_proxy,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)


@configclass
class TQBot2StandardWalkRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """TQBot2 adaptation of the repository's standard rough-terrain walker."""

    base_link_name = "trunk"
    foot_link_name = ".*_calf"
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        # Scene and robot-specific link names.
        self.scene.robot = TQBOT2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.spawn.func = spawn_tqbot2_walk_with_leg_collisions
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # Nominal slope is rise/run. Retain the level curriculum and raise its ceiling to 30 degrees.
        for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
            self.scene.terrain.terrain_generator.sub_terrains[name].slope_range = (0.0, math.tan(math.radians(30.0)))
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        total = sum(t.proportion for t in terrains.values())
        for terrain in terrains.values():
            terrain.proportion = 0.9 * terrain.proportion / total
        terrains["high_friction_flat"] = MeshPlaneTerrainCfg(proportion=0.1)

        # Match the existing TQBot2 velocity-tracking observation interface.
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # Keep one continuous, small-amplitude action mapping in every state.
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        # JointPositionAction clips processed joint targets, preserving raw-action observations.
        self.actions.joint_pos.clip = {
            ".*_thigh_joint": (0.6, 3.2),
            "^(?!.*_thigh_joint).*": (-100.0, 100.0),
        }
        self.actions.joint_pos.joint_names = self.joint_names

        # Domain randomization, with the TQBot2 trunk selected explicitly.
        # Apply the foot material override after the inherited whole-body event.
        self.events.randomize_foot_material = self.events.randomize_rigid_body_material.copy()
        foot_material = self.events.randomize_foot_material.params
        foot_material["asset_cfg"].body_names = [self.foot_link_name]
        foot_material["static_friction_range"] = (0.15, 1.0)
        foot_material["dynamic_friction_range"] = (0.15, 0.8)
        # Last startup material event: fixed 2.5 overrides the usual foot ranges
        # only on the dedicated flat columns. Ground remains 1.0 (multiply).
        self.events.high_friction_flat_material = self.events.randomize_foot_material.copy()
        self.events.high_friction_flat_material.func = HighFrictionFlatMaterial
        self.events.high_friction_flat_material.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
            "terrain_name": "high_friction_flat", "friction": 2.5,
        }
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2),
                "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # Standard velocity-tracking reward structure.
        self.rewards.is_terminated.weight = 0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        foot_motion_params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "command_name": "base_velocity", "foot_offset": (0.0, 0.0, -0.153),
            "foot_radius": 0.012, "target_clearance": 0.03,
        }
        self.rewards.feet_slide = RewTerm(
            func=FootGroundMotion, weight=-0.5, params={**foot_motion_params, "mode": "slip"},
        )
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_swing_clearance = RewTerm(
            func=FootGroundMotion, weight=-0.5, params={**foot_motion_params, "mode": "clearance"},
        )
        self.rewards.feet_gait.weight = 0.25
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_calf", "RR_calf"), ("FR_calf", "RL_calf"))
        self.rewards.upward.weight = 1.0

        # Normalized costs fade out once upright and clear of trunk-ground contact.
        recovery_params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]),
        }
        self.rewards.recovery_effort = RewTerm(
            func=recovery_joint_effort, weight=-0.5, params=recovery_params.copy(),
        )
        self.rewards.recovery_torque_change = RewTerm(
            func=RecoveryTorqueChange, weight=-0.25,
            params={**recovery_params, "reference_dt": 0.02},
        )
        self.rewards.recovery_joint_velocity = RewTerm(
            func=recovery_joint_velocity, weight=-0.25,
            params={**recovery_params, "reference_speed": 8.0},
        )
        self.rewards.walking_energy_efficiency = RewTerm(
            func=walking_energy_efficiency, weight=0.5,
            params={**recovery_params, "command_name": "base_velocity", "energy_per_metre": 500.0,
                    "speed_floor": 0.1, "yaw_radius": 0.25, "tracking_std": 0.5},
        )
        self.rewards.walking_joule_proxy = RewTerm(
            func=walking_joule_proxy, weight=-0.15, params=recovery_params.copy(),
        )

        self.disable_zero_weight_rewards()
        self.terminations.illegal_contact = None
        self.terminations.high_friction_flat_exit = DoneTerm(func=high_friction_flat_exit)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
