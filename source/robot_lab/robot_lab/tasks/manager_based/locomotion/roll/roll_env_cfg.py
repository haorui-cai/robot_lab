"""Base configuration for rolling locomotion environments."""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.roll.mdp as roll_mdp
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as vmdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class RollSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a rolling robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class RollCommandsCfg:
    """Command specifications for the rolling MDP."""

    base_velocity = vmdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=vmdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class RollActionsCfg:
    """Action specifications for the rolling MDP."""

    joint_pos = vmdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class RollObservationsCfg:
    """Observation specifications for the rolling MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=vmdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=vmdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=vmdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(func=vmdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=vmdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)}, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=vmdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)}, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0), scale=1.0)
        actions = ObsTerm(func=vmdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        height_scan = ObsTerm(func=vmdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-1.0, 1.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=vmdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=vmdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=vmdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(func=vmdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=vmdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)}, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=vmdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)}, clip=(-100.0, 100.0), scale=1.0)
        actions = ObsTerm(func=vmdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        height_scan = ObsTerm(func=vmdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RollEventCfg:
    """Configuration for events."""

    randomize_rigid_body_material = EventTerm(
        func=vmdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.3, 1.0), "dynamic_friction_range": (0.3, 0.8), "restitution_range": (0.0, 0.5), "num_buckets": 64},
    )
    randomize_rigid_body_mass_base = EventTerm(
        func=vmdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "mass_distribution_params": (-3.0, 3.0), "operation": "add", "recompute_inertia": True},
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=vmdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.5, 1.5), "operation": "scale", "recompute_inertia": True},
    )
    randomize_com_positions = EventTerm(
        func=vmdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}},
    )
    randomize_apply_external_force_torque = EventTerm(
        func=vmdp.apply_external_force_torque, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "force_range": (-10.0, 10.0), "torque_range": (-10.0, 10.0)},
    )
    randomize_reset_joints = EventTerm(
        func=vmdp.reset_joints_by_scale, mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    randomize_actuator_gains = EventTerm(
        func=vmdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stiffness_distribution_params": (0.5, 2.0), "damping_distribution_params": (0.5, 2.0), "operation": "scale", "distribution": "uniform"},
    )
    randomize_reset_base = EventTerm(
        func=vmdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
    )
    randomize_push_robot = EventTerm(
        func=vmdp.push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RollRewardsCfg:
    """Reward terms for the rolling MDP."""

    is_terminated = RewTerm(func=vmdp.is_terminated, weight=0.0)
    lin_vel_z_l2 = RewTerm(func=vmdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=vmdp.ang_vel_xy_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=vmdp.flat_orientation_l2, weight=0.0)
    base_height_l2 = RewTerm(func=vmdp.base_height_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "sensor_cfg": SceneEntityCfg("height_scanner_base"), "target_height": 0.0})
    body_lin_acc_l2 = RewTerm(func=vmdp.body_lin_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="")})
    joint_torques_l2 = RewTerm(func=vmdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    joint_vel_l2 = RewTerm(func=vmdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    joint_acc_l2 = RewTerm(func=vmdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    joint_pos_limits = RewTerm(func=vmdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    joint_vel_limits = RewTerm(func=vmdp.joint_vel_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0})
    joint_power = RewTerm(func=vmdp.joint_power, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    stand_still = RewTerm(func=vmdp.stand_still, weight=0.0, params={"command_name": "base_velocity", "command_threshold": 0.1, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    joint_pos_penalty = RewTerm(func=vmdp.joint_pos_penalty, weight=0.0, params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stand_still_scale": 5.0, "velocity_threshold": 0.5, "command_threshold": 0.1})
    joint_mirror = RewTerm(func=vmdp.joint_mirror, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot"), "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]]})
    action_rate_l2 = RewTerm(func=vmdp.action_rate_l2, weight=0.0)
    undesired_contacts = RewTerm(func=vmdp.undesired_contacts, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0})
    contact_forces = RewTerm(func=vmdp.contact_forces, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 100.0})
    track_lin_vel_xy_exp = RewTerm(func=vmdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z_exp = RewTerm(func=vmdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    feet_air_time = RewTerm(func=vmdp.feet_air_time, weight=0.0, params={"command_name": "base_velocity", "threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="")})
    feet_air_time_variance = RewTerm(func=vmdp.feet_air_time_variance_penalty, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")})
    feet_gait = RewTerm(func=vmdp.GaitReward, weight=0.0, params={"std": math.sqrt(0.5), "command_name": "base_velocity", "max_err": 0.2, "velocity_threshold": 0.5, "command_threshold": 0.1, "synced_feet_pair_names": (("", ""), ("", "")), "asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("contact_forces")})
    feet_contact = RewTerm(func=vmdp.feet_contact, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "command_name": "base_velocity", "expect_contact_num": 2})
    feet_contact_without_cmd = RewTerm(func=vmdp.feet_contact_without_cmd, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "command_name": "base_velocity"})
    feet_stumble = RewTerm(func=vmdp.feet_stumble, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")})
    feet_slide = RewTerm(func=vmdp.feet_slide, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "asset_cfg": SceneEntityCfg("robot", body_names="")})
    feet_height = RewTerm(func=vmdp.feet_height, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "tanh_mult": 2.0, "target_height": 0.05, "command_name": "base_velocity"})
    feet_height_body = RewTerm(func=vmdp.feet_height_body, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "tanh_mult": 2.0, "target_height": -0.3, "command_name": "base_velocity"})
    upward = RewTerm(func=vmdp.upward, weight=0.0)

    # Roll-specific rewards
    roll_progress = RewTerm(func=roll_mdp.roll_progress, weight=0.0, params={"target_speed": 0.9})
    roll_rate = RewTerm(func=roll_mdp.roll_rate, weight=0.0, params={"target_roll_rate": 4.5})
    roll_attitude = RewTerm(func=roll_mdp.roll_attitude, weight=0.0, params={"target_roll": 4.5})
    base_height_above = RewTerm(func=roll_mdp.base_height_above_l1, weight=0.0, params={"threshold": 0.18, "asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("height_scanner_base")})
    roll_anti_jump = RewTerm(func=roll_mdp.roll_anti_jump, weight=0.0, params={"vel_weight": 2.0, "acc_weight": 5.0e-3})
    roll_contact_bonus = RewTerm(func=roll_mdp.roll_contact_bonus, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", "base"])})


@configclass
class RollTerminationsCfg:
    """Termination terms for the rolling MDP."""

    time_out = DoneTerm(func=vmdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(func=vmdp.terrain_out_of_bounds, params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0}, time_out=True)
    illegal_contact = DoneTerm(func=vmdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0})


@configclass
class RollCurriculumCfg:
    """Curriculum terms for the rolling MDP."""

    terrain_levels = CurrTerm(func=vmdp.terrain_levels_vel)
    command_levels = CurrTerm(func=vmdp.command_levels_lin_vel)


@configclass
class RollRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the rolling locomotion environment."""

    scene: RollSceneCfg = RollSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RollObservationsCfg = RollObservationsCfg()
    actions: RollActionsCfg = RollActionsCfg()
    commands: RollCommandsCfg = RollCommandsCfg()
    rewards: RollRewardsCfg = RollRewardsCfg()
    terminations: RollTerminationsCfg = RollTerminationsCfg()
    events: RollEventCfg = RollEventCfg()
    curriculum: RollCurriculumCfg = RollCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr is not None and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)
