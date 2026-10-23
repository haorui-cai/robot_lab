"""Assisted-start rolling with performance-driven phase and yaw curricula."""

from isaaclab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.roll.mdp import hip_only_cycle as mdp
from robot_lab.tasks.manager_based.locomotion.roll.mdp.base_acceleration import base_acceleration_rewards

from .hip_only_env_cfg import TQBot2RollHipOnlyEnvCfg


@configclass
class CycleCommandsCfg:
    base_velocity = mdp.RollYawCommandCfg()


@configclass
class CycleRewardsCfg:
    roll_cycle_tracking = RewardTermCfg(func=mdp.cycle_roll_reward, weight=8.0)
    yaw_cycle_tracking = RewardTermCfg(func=mdp.cycle_yaw_reward, weight=2.0)
    base_linear_acceleration = base_acceleration_rewards()["base_linear_acceleration"]
    base_angular_acceleration = base_acceleration_rewards()["base_angular_acceleration"]


@configclass
class CycleCurriculumCfg:
    rolling = CurriculumTermCfg(func=mdp.rolling_curriculum)


@configclass
class TQBot2RollHipOnlyCycleEnvCfg(TQBot2RollHipOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = {".*R_hip_joint": 1.52, ".*L_hip_joint": -1.52}
        self.actions.joint_pos.scale = 1.0
        self.actions.joint_pos.offset = {".*R_hip_joint": 0.52, ".*L_hip_joint": -0.52}
        # Clip the resulting targets as well as PPO's raw [-1, 1] actions.
        self.actions.joint_pos.clip = {".*R_hip_joint": (-0.48, 1.52), ".*L_hip_joint": (-1.52, 0.48)}
        self.commands = CycleCommandsCfg()
        self.rewards = CycleRewardsCfg()
        self.curriculum = CycleCurriculumCfg()
        self.events.randomize_reset_base.func = mdp.reset_rolling_phase
        self.events.randomize_reset_base.params = {}
        self.events.randomize_push_robot = None
        self.terminations.roll_too_slow = None
        self.terminations.cycle_stalled = TerminationTermCfg(func=mdp.cycle_stalled)
        self.terminations.invalid_rolling_pose = TerminationTermCfg(func=mdp.invalid_rolling_pose)
        for group in (self.observations.policy, self.observations.critic):
            group.rolling_cycle = ObservationTermCfg(func=mdp.cycle_observation)
            group.support_contacts = ObservationTermCfg(func=mdp.support_observation)
