"""Standard velocity-tracking walking task for TQBot2.

This task deliberately uses the repository's regular joint-position action
term.  Recovery and rolling remain separate skills so that their specialised
action mappings do not alter the walking policy's MDP.
"""

import math
from pathlib import Path

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from robot_lab.assets.tqbot import TQBOT2_WALK_CFG
from robot_lab.assets.tqbot_collision import spawn_tqbot2_walk_with_leg_collisions
from robot_lab.assets.tqbot_walk_stance import (
    WALK_LEGACY_POLICY_ZERO_JOINT_POS, WALK_THIGH_SOFT_LIMITS,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.events import set_soft_joint_position_limits
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.foot_ground import SupportFootSlip, SwingFootClearance
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.frequency_excess import FootFrequencyExcess
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations import JointPositionRelativeToReference
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.normalized_gait import FootCadenceReward, PeriodNormalizedGaitReward
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import FootDisplacementSymmetry
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.swing_posture import SwingJointPositionPenalty
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terrain_statistics import (
    terrain_levels_with_statistics, terrain_level_statistics,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)


@configclass
class TQBot2StandardWalkRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """TQBot2 adaptation of the repository's standard rough-terrain walker."""

    base_link_name = "trunk"
    foot_link_name = ".*_foot"
    training_torque_scale = 0.90
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        # Scene and robot-specific link names.
        self.scene.robot = TQBOT2_WALK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.asset_path = str(Path(TQBOT2_WALK_CFG.spawn.asset_path).with_name("TQBot2_walk.urdf"))
        # Set walking PD gains and the training torque margin on this task's
        # implicit actuator copies, leaving the shared robot asset unchanged.
        for name, actuator in self.scene.robot.actuators.items():
            self.scene.robot.actuators[name] = actuator.replace(
                stiffness=20.0,
                damping=0.5,
                effort_limit_sim=actuator.effort_limit_sim * self.training_torque_scale,
            )
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.spawn.func = spawn_tqbot2_walk_with_leg_collisions
        self.events.walk_thigh_soft_limits = EventTerm(
            func=set_soft_joint_position_limits, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint"),
                    "lower": WALK_THIGH_SOFT_LIMITS[0], "upper": WALK_THIGH_SOFT_LIMITS[1]},
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # Nominal slope is rise/run. Retain the level curriculum and raise its ceiling to 30 degrees.
        for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
            self.scene.terrain.terrain_generator.sub_terrains[name].slope_range = (0.0, math.tan(math.radians(30.0)))
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        # Match the shorter-leg step geometry used by TQBot2RoughEnvCfg.
        terrains["pyramid_stairs"].step_height_range = (0.035, 0.161)
        terrains["pyramid_stairs_inv"].step_height_range = (0.035, 0.161)
        terrains["boxes"].grid_height_range = (0.035, 0.14)

        # Match the existing TQBot2 velocity-tracking observation interface.
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # Actor and critic use policy coordinates, independently of standing
        # rewards. Each term resolves the reference in its own joint order.
        for observations in (self.observations.policy, self.observations.critic):
            observations.joint_pos.func = JointPositionRelativeToReference
            observations.joint_pos.params["reference_joint_pos"] = WALK_LEGACY_POLICY_ZERO_JOINT_POS.copy()

        # Keep one continuous, small-amplitude action mapping in every state.
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names
        # Action zero is independent of the articulation's standing/reset pose.
        # Use the legacy 0.80-rad thigh reference; retain the 1.35-rad reference
        # in tqbot_walk_stance.py for separate experiments.
        self.actions.joint_pos.use_default_offset = False
        self.actions.joint_pos.offset = WALK_LEGACY_POLICY_ZERO_JOINT_POS.copy()

        # Domain randomization, with the TQBot2 trunk selected explicitly.
        # Apply the foot material override after the inherited whole-body event.
        self.events.randomize_foot_material = self.events.randomize_rigid_body_material.copy()
        foot_material = self.events.randomize_foot_material.params
        foot_material["asset_cfg"].body_names = [self.foot_link_name]
        foot_material["static_friction_range"] = (0.3, 1.0)
        foot_material["dynamic_friction_range"] = (0.3, 0.8)
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
        # Both penalties read default_joint_pos, which remains the standing
        # reference (thigh=0.80), independently of the policy action offset.
        self.rewards.stand_still.weight = -2.0
        # Keep 20% of the original posture penalty. Exclude all three joints
        # of commanded swing legs; support and stop-command errors retain
        # their original norm and standing multiplier before this weight.
        self.rewards.joint_pos_penalty.weight = -0.2
        self.rewards.joint_pos_penalty.func = SwingJointPositionPenalty
        self.rewards.joint_pos_penalty.params.update(
            sensor_cfg=SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            foot_joint_names={
                f"{leg}_foot": tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf"))
                for leg in ("FR", "FL", "RR", "RL")
            },
            swing_scale=0.0,
        )
        # Retain the historical log key and weight, but compare diagonal foot
        # X/Z displacement from standing instead of raw joint-angle equality.
        self.rewards.joint_mirror.func = FootDisplacementSymmetry
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_pairs": (("FR_foot", "RL_foot"), ("FL_foot", "RR_foot")),
            "calibration_perturbation_rad": 0.05,
            "calibration_samples": 8192,
            "calibration_seed": 42,
        }
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.undesired_contacts.weight = -1.0
        # Allow foot, calf and thigh contact without disabling physical collisions.
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [r"^(?!.*_(?:foot|calf|thigh)$).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        # A complete same-foot touchdown cycle should take 0.5 s. Score the
        # measured frequency each control step, rather than paying per landing.
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_frequency = RewTerm(
            func=FootCadenceReward, weight=0.2,
            params={
                "target_frequency": 2.0, "frequency_scale": 0.5,
                "command_name": "base_velocity", "command_threshold": 0.1,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
                "period_range": (0.08, 2.0), "stale_periods": 2.0,
            },
        )
        # Preserve all four 5 ms substeps for the one-sided excess cost. A
        # separate foot-only sensor leaves existing contact rewards unchanged.
        self.scene.foot_cycle_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot", update_period=self.sim.dt,
            history_length=self.decimation, track_air_time=False,
        )
        self.rewards.feet_frequency_excess = RewTerm(
            func=FootFrequencyExcess, weight=-0.5,
            params={
                "threshold_frequency": 2.0, "frequency_scale": 2.0,
                "command_name": "base_velocity", "command_threshold": 0.1,
                "confirmation_time": 0.01, "force_threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("foot_cycle_contacts", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.func = SupportFootSlip
        self.rewards.feet_slide.params["foot_radius"] = 0.012
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0.0
        # Measure the actual foot sphere's bottom against the local terrain
        # plane. Only penalize shortfall below 5 cm; higher feet have zero cost.
        # Normalize by 5 cm: -0.5 / 0.05**2 retains 40 times the original
        # -5 metre-squared cost for feet below the target.
        self.rewards.feet_ground_clearance = RewTerm(
            func=SwingFootClearance, weight=-0.5,
            params={
                "command_name": "base_velocity", "target_clearance": 0.05,
                "error_scale": 0.05,
                "tanh_mult": 2.0, "foot_radius": 0.012,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            },
        )
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.feet_gait.func = PeriodNormalizedGaitReward
        gait_params = self.rewards.feet_gait.params
        # Keep the numeric shaping constants, now applied to fractions of a
        # completed cycle rather than seconds. Bounds reject unusable period
        # estimates; they do not command a preferred gait frequency.
        gait_params["phase_error_scale"] = gait_params.pop("std")
        gait_params["max_phase_error"] = gait_params.pop("max_err")
        gait_params["period_range"] = (0.08, 2.0)
        gait_params["stale_periods"] = 2.0
        self.rewards.upward.weight = 1.0

        self.disable_zero_weight_rewards()
        # The zero-weight cleanup expects reward configs; disable these afterwards.
        self.rewards.recovery_effort = None
        self.rewards.recovery_torque_change = None
        self.rewards.recovery_joint_velocity = None
        self.rewards.feet_swing_clearance = None
        self.rewards.walking_energy_efficiency = None
        self.rewards.walking_joule_proxy = None
        self.terminations.illegal_contact = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        # Preserve the original total metric and log the independently evolving populations.
        self.curriculum.terrain_levels.func = terrain_levels_with_statistics
        self.curriculum.terrain_breakdown = CurrTerm(func=terrain_level_statistics)
