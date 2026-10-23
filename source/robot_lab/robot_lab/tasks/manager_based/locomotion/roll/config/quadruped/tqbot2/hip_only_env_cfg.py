"""Rolling with four hip actions and physically constrained curled legs."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.tasks.manager_based.locomotion.roll.mdp.events import reset_hips_to_default

from .rough_env_cfg import TQBot2RollRoughEnvCfg


@configclass
class TQBot2RollHipOnlyEnvCfg(TQBot2RollRoughEnvCfg):
    """Keep the existing rolling task, but let PPO control only the hips."""

    def __post_init__(self):
        super().__post_init__()
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.2, 2.2, 1.4)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        hip_names = [name for name in self.joint_names if "_hip_" in name]
        # Fixed joint transforms bake in thigh=1.68 and calf=-2.82 rad.
        # Merge each fixed chain, including its mass/inertia and collision shapes.
        self.scene.robot.spawn.asset_path = (
            f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/TQBot/TQBot2_description/urdf/TQBot2_hip_only.urdf"
        )
        self.scene.robot.init_state.joint_pos = {
            ".*R_hip_joint": 1.57,
            ".*L_hip_joint": -1.14,
        }
        hip_actuator = self.scene.robot.actuators["hip_thigh"].replace(joint_names_expr=[".*_hip_joint"])
        self.scene.robot.actuators = {"hip": hip_actuator}
        self.actions.joint_pos.joint_names = hip_names
        self.actions.joint_pos.scale = 0.50
        self.actions.joint_pos.offset = {".*R_hip_joint": 1.57, ".*L_hip_joint": -1.14}

        # TQBot2's LEFT_SUPPORT phase window is (-3.0, -1.6) rad.
        # At roll=-2 and these hip angles, the FL/RL shell minima are
        # -0.110824 m relative to trunk. Spawn 2.18 mm above the plane.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.113)
        self.events.randomize_reset_base.params = {
            "pose_range": {"roll": (-2.0, -2.0)},
            "velocity_range": {},
        }
        self.events.randomize_reset_joints.func = reset_hips_to_default
        self.events.randomize_reset_joints.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=hip_names)}
        self.events.randomize_apply_external_force_torque = None
        self.commands.base_velocity.debug_vis = True

        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["asset_cfg"].joint_names = hip_names
            group.joint_vel.params["asset_cfg"].joint_names = hip_names
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=hip_names)

        # Only the four active joints contribute to the joint penalties.
        for name in ("joint_torques_l2", "joint_acc_l2", "joint_pos_limits"):
            getattr(self.rewards, name).params["asset_cfg"] = SceneEntityCfg("robot", joint_names=hip_names)
        self.rewards.roll_thigh_jitter = None
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_hip_joint", "RL_hip_joint"],
            ["FR_hip_joint", "RR_hip_joint"],
        ]
        self.disable_zero_weight_rewards()
