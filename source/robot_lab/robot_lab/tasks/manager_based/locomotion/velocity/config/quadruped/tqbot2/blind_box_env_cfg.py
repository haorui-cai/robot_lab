"""Dedicated manually selected climbing skill with a proprioceptive history actor."""

import xml.etree.ElementTree as ET

from isaaclab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg

from . import blind_box_mdp as box
from .rough_env_cfg import TQBot2RoughEnvCfg


@configclass
class TQBot2BlindBoxEnvCfg(TQBot2RoughEnvCfg):
    box_evaluation_height: float | None = None
    box_evaluation_length: float | None = None
    box_curriculum_resume_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.episode_length_s = 10.0
        self.scene.terrain.class_type = box.BlindBoxTerrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.restitution = 0.0
        self.scene.height_scanner_base = None
        self.scene.height_scanner.pattern_cfg.size = [2.4, 1.0]

        self.observations.policy.history_length = 12
        self.observations.policy.flatten_history_dim = True
        joints = SceneEntityCfg("robot", joint_names=self.joint_names, preserve_order=True)
        self.observations.policy.joint_tracking_error = ObservationTermCfg(
            func=box.joint_tracking_error, params={"asset_cfg": joints}, clip=(-3.0, 3.0),
            noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        self.observations.critic.box_geometry = ObservationTermCfg(func=box.box_geometry_observation)
        self.observations.critic.joint_pos.params["asset_cfg"] = joints
        self.observations.critic.joint_vel.params["asset_cfg"] = joints

        # Joint-position mapping remains continuous; target clipping uses physical limits.
        xml = ET.parse(self.scene.robot.spawn.asset_path).getroot()
        self.actions.joint_pos.clip = {}
        for name in self.joint_names:
            limits = xml.find(f"joint[@name='{name}']/limit")
            low, high = float(limits.get("lower")), float(limits.get("upper"))
            mid, half = (low + high) / 2, (high - low) * 0.45
            self.actions.joint_pos.clip[name] = (mid - half, mid + half)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.randomize_reset_base.func = box.reset_box
        self.events.randomize_reset_base.params = {}
        self.events.randomize_reset_joints.params["position_range"] = (0.9, 1.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.1, 0.1)
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.3, 0.3)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.randomize_com_positions.params["com_range"] = {axis: (-0.01, 0.01) for axis in ("x", "y", "z")}
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_rigid_body_material.params.update(
            static_friction_range=(0.6, 1.1), dynamic_friction_range=(0.6, 1.0), restitution_range=(0.0, 0.0),
        )

        for name in ("feet_gait", "joint_mirror", "feet_air_time", "feet_air_time_variance",
                     "feet_height_body", "feet_contact_without_cmd", "stand_still", "upward",
                     "track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            getattr(self.rewards, name).weight = 0
        self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.01
        self.rewards.joint_pos_penalty.weight = -0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["trunk"]
        self.rewards.feet_slide.weight = 0  # inherited term measures calf origins, not foot centers
        self.rewards.box_progress = RewardTermCfg(func=box.progress_reward, weight=3.0)
        self.rewards.box_goal = RewardTermCfg(func=box.goal_reward, weight=1.0)
        self.rewards.box_top_support = RewardTermCfg(func=box.top_support_reward, weight=2.0)
        # Isaac Lab multiplies all reward terms by dt: these yield +5 / -5 per terminal event.
        self.rewards.box_success = RewardTermCfg(func=box.success_reward, weight=250.0)
        self.rewards.box_failure = RewardTermCfg(func=box.failure_reward, weight=-250.0)
        self.disable_zero_weight_rewards()
        self.terminations.terrain_out_of_bounds = None
        self.terminations.box_success = TerminationTermCfg(func=box.box_success)
        self.terminations.box_failure = TerminationTermCfg(func=box.box_failure)
        self.curriculum.terrain_levels = None
        self.curriculum.box_height = CurriculumTermCfg(func=box.box_curriculum)
