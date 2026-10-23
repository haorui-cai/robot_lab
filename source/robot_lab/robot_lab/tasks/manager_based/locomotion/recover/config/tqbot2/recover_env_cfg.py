# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""TQBot2: arbitrary fallen orientation -> upright body on the ground."""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_lab.assets.tqbot import TQBOT2_CFG
from robot_lab.tasks.manager_based.locomotion.recover import mdp as recover_mdp

JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
# Neutral action/observation reference only, NOT a recovery objective.
# unitree_guide/include/FSM/State_FixedLie.h, FR / FL / RR / RL.
PRONE_JOINT_POS = {".*_hip_joint": 0.0, ".*_thigh_joint": 1.36, ".*_calf_joint": -2.65}


@configclass
class RecoverSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot = TQBOT2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=TQBOT2_CFG.spawn.replace(func=recover_mdp.spawn_with_cross_leg_collisions, self_collision=True),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        use_default_offset=True,
        scale={".*_hip_joint": 1.588, ".*_thigh_joint": 4.75, ".*_calf_joint": 1.735},
        # Isaac clips the scaled position target, not the raw policy action.
        clip={".*_hip_joint": (-1.588, 1.588), ".*_thigh_joint": (-3.141, 6.108), ".*_calf_joint": (-2.975, -0.916)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-20.0, 20.0), noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-30.0, 30.0),
            scale=0.1,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_height = ObsTerm(func=recover_mdp.base_height)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    reset_robot = EventTerm(func=recover_mdp.reset_supine, mode="reset")


@configclass
class RewardsCfg:
    upright = RewTerm(func=recover_mdp.upright, weight=3.0)
    prone_settled = RewTerm(func=recover_mdp.prone_settled, weight=6.0)
    legs_down = RewTerm(func=recover_mdp.upright_legs_down, weight=4.0)
    # Keep scoring the maintained pose until the dwell-time termination fires.
    prone = RewTerm(func=recover_mdp.is_prone, weight=4.0)
    # At 50 Hz this is +50 once. It exceeds the maximum discounted continuing
    # posture return (17 * .02 / (1 - .99) = 34), so success is worth finishing.
    success_bonus = RewTerm(func=mdp.is_terminated_term, weight=2500.0, params={"term_keys": "success"})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    hold_joint_velocity = RewTerm(func=recover_mdp.hold_joint_velocity, weight=-0.3)
    hold_joint_acceleration = RewTerm(func=recover_mdp.hold_joint_acceleration, weight=-2.0e-5)
    hold_target_rate = RewTerm(func=recover_mdp.HoldTargetRate, weight=-0.01)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=recover_mdp.StableProne, params={"hold_time": 4.0, "settle_time": 1.0})
    escaped = DoneTerm(func=recover_mdp.escaped)
    # Body/hip/thigh/knee contacts and inverted orientations are valid states.


@configclass
class TQBot2RecoverEnvCfg(ManagerBasedRLEnvCfg):
    scene: RecoverSceneCfg = RecoverSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 16.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.096)
        self.scene.robot.init_state.rot = (0.0, 1.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = PRONE_JOINT_POS.copy()
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        # Use the shared TQBot2 deployment-aligned nominal gains (14 / 0.6).
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 2
        self.viewer.eye = (2.5, 2.5, 1.7)
        self.viewer.lookat = (0.0, 0.0, 0.1)


@configclass
class TQBot2RecoverEnvCfg_PLAY(TQBot2RecoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
