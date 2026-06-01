"""MDP functions for rolling locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def roll_rate(
    env: ManagerBasedRLEnv, std: float, command_name: str, roll_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for tracking commanded roll angular velocity around x-axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll_vel = asset.data.root_ang_vel_b[:, 0]
    cmd_roll = env.command_manager.get_command(command_name)[:, 1] / roll_radius
    error = torch.square(roll_vel - cmd_roll)
    return torch.exp(-error / std ** 2)


def roll_attitude(
    env: ManagerBasedRLEnv, target_roll: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty for pitch/yaw deviation and roll rate deviation from target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll = asset.data.root_ang_vel_b[:, 0]
    pitch = asset.data.root_ang_vel_b[:, 1]
    yaw = asset.data.root_ang_vel_b[:, 2]
    return -(roll - target_roll) ** 2 - pitch**2 - yaw**2


def roll_progress(
    env: ManagerBasedRLEnv, target_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for forward progress along x-axis at target speed."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    error = lin_vel_x - target_speed
    return torch.exp(-(error**2) / (target_speed * 0.5) ** 2)


def base_height_above_l1(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_base"),
) -> torch.Tensor:
    """L1 penalty when base height exceeds threshold to prevent jumping."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.clamp(base_height - threshold, min=0.0)


def roll_anti_jump(
    env: ManagerBasedRLEnv,
    vel_weight: float = 2.0,
    acc_weight: float = 5.0e-3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Combined penalty on vertical velocity and acceleration."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_z = asset.data.root_lin_vel_b[:, 2]
    lin_acc_z = asset.data.root_lin_acc_b[:, 2] if hasattr(asset.data, "root_lin_acc_b") else torch.zeros_like(lin_vel_z)
    return vel_weight * (lin_vel_z**2) + acc_weight * (lin_acc_z**2)


def roll_contact_bonus(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*"),
) -> torch.Tensor:
    """Reward for body and leg ground contact using max force per env."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_force = torch.max(torch.norm(net_forces, dim=-1).max(dim=1).values, dim=-1).values
    return torch.tanh(max_force / 50.0)


def roll_hip_asymmetry(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_thigh"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward hip asymmetry when thighs are pressing against the ground during rolling."""
    # Thigh contact gate: how hard thighs are pushing
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_thigh_force = torch.max(torch.norm(net_forces, dim=-1).max(dim=1).values, dim=-1).values
    contact_gate = torch.tanh(max_thigh_force / 10.0)

    # Hip asymmetry: left vs right hip joint position difference
    asset: Articulation = env.scene[asset_cfg.name]
    l_hip_ids = asset.find_joints(".*L_hip_joint")[0]
    r_hip_ids = asset.find_joints(".*R_hip_joint")[0]
    left_hip = torch.mean(asset.data.joint_pos[:, l_hip_ids], dim=1)
    right_hip = torch.mean(asset.data.joint_pos[:, r_hip_ids], dim=1)
    hip_diff = torch.abs(left_hip - right_hip)

    return contact_gate * hip_diff


def roll_thigh_drive(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_thigh"),
    max_duration: float = 0.1,
):
    """Reward thigh-ground force, but only for the first max_duration seconds per contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_thigh_force = torch.max(torch.norm(net_forces, dim=-1).max(dim=1).values, dim=-1).values

    # Initialize contact timer on env if not exists
    if not hasattr(env, "roll_thigh_contact_timer"):
        env.roll_thigh_contact_timer = torch.zeros(env.num_envs, device=env.device)

    # Detect if currently in contact (force > 25N threshold for meaningful push)
    in_contact = max_thigh_force > 25.0

    # Increment timer for envs in contact, reset for envs not in contact
    step_dt = env.cfg.decimation * env.cfg.sim.dt
    env.roll_thigh_contact_timer = torch.where(
        in_contact,
        env.roll_thigh_contact_timer + step_dt,
        torch.zeros_like(env.roll_thigh_contact_timer),
    )

    # Only reward when within the time window
    within_window = env.roll_thigh_contact_timer <= max_duration
    reward = torch.where(within_window, torch.tanh(max_thigh_force / 50.0), torch.zeros_like(max_thigh_force))
    return reward


def roll_curled_posture(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining a curled posture (legs tucked in) for efficient rolling."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos

    # Right hips target +1.58, left hips target -1.58, thighs target 1.67
    r_hip_ids = asset.find_joints(".*R_hip_joint")[0]
    l_hip_ids = asset.find_joints(".*L_hip_joint")[0]
    thigh_ids = asset.find_joints(".*_thigh_joint")[0]

    reward = torch.zeros(env.num_envs, device=env.device)
    reward += torch.exp(-torch.square(joint_pos[:, r_hip_ids] - 1.58)).mean(dim=1)
    reward += torch.exp(-torch.square(joint_pos[:, l_hip_ids] + 1.58)).mean(dim=1)
    reward += torch.exp(-torch.square(joint_pos[:, thigh_ids] - 1.67)).mean(dim=1)
    reward /= 3.0
    return reward


def roll_yaw_rate(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of yaw velocity command without gravity projection decay."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / std ** 2)
