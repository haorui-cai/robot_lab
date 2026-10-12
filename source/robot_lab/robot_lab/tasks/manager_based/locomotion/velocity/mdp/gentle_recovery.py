"""Recovery-only effort costs for the walking policy."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .recovery_costs import (
    TorqueChangeHistory, distance_energy_score, normalized_effort, normalized_velocity, recovery_gate,
)


def _gate(env, asset_cfg, sensor_cfg):
    robot = env.scene[asset_cfg.name]
    forces = env.scene[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    trunk_force = forces.norm(dim=-1).amax(dim=(1, 2))
    return recovery_gate(-robot.data.projected_gravity_b[:, 2], trunk_force)


def recovery_joint_effort(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    joints = asset_cfg.joint_ids
    cost = normalized_effort(robot.data.applied_torque[:, joints], robot.data.joint_effort_limits[:, joints])
    return cost * _gate(env, asset_cfg, sensor_cfg)


def recovery_joint_velocity(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, reference_speed: float):
    robot = env.scene[asset_cfg.name]
    cost = normalized_velocity(robot.data.joint_vel[:, asset_cfg.joint_ids], reference_speed)
    return cost * _gate(env, asset_cfg, sensor_cfg)


class RecoveryTorqueChange(ManagerTermBase):
    """Penalize torque changes per 20 ms, with independent history for each episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.robot = env.scene[self.asset_cfg.name]
        self.history = TorqueChangeHistory(self.robot.data.applied_torque[:, self.asset_cfg.joint_ids])

    def reset(self, env_ids=None):
        self.history.reset(env_ids)

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, reference_dt: float):
        joints = asset_cfg.joint_ids
        cost = self.history.compute(
            self.robot.data.applied_torque[:, joints], self.robot.data.joint_effort_limits[:, joints],
            env.step_dt, reference_dt,
        )
        return cost * _gate(env, asset_cfg, sensor_cfg)


def walking_energy_efficiency(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, command_name: str,
    energy_per_metre: float, speed_floor: float, yaw_radius: float, tracking_std: float,
):
    """Adaptive distance-normalized work reward, conditioned on following the command.

    Absolute joint work is a mechanical proxy; no battery efficiency or regeneration model is assumed.
    See Liang et al., Adaptive Energy Regularization, arXiv:2403.20001v2.
    """
    robot = env.scene[asset_cfg.name]
    joints = asset_cfg.joint_ids
    command = env.command_manager.get_command(command_name)
    power = (robot.data.applied_torque[:, joints] * robot.data.joint_vel[:, joints]).abs().sum(dim=1)
    motion = robot.data.root_lin_vel_b[:, :2].norm(dim=1) + yaw_radius * robot.data.root_ang_vel_b[:, 2].abs()
    requested_motion = command[:, :2].norm(dim=1) + yaw_radius * command[:, 2].abs()
    error = (robot.data.root_lin_vel_b[:, :2] - command[:, :2]).square().sum(dim=1)
    error += (robot.data.root_ang_vel_b[:, 2] - command[:, 2]).square()
    tracking = torch.exp(-error / tracking_std**2)
    score = distance_energy_score(power, motion, tracking, requested_motion > 0.1, energy_per_metre, speed_floor)
    return score * (1.0 - _gate(env, asset_cfg, sensor_cfg))


def walking_joule_proxy(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    """Normalized squared torque as an uncalibrated copper-loss proxy, including static support."""
    robot = env.scene[asset_cfg.name]
    joints = asset_cfg.joint_ids
    cost = normalized_effort(robot.data.applied_torque[:, joints], robot.data.joint_effort_limits[:, joints])
    return cost * (1.0 - _gate(env, asset_cfg, sensor_cfg))
