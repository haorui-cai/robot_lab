# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MDP functions for rolling locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def roll_rate(
    env: ManagerBasedRLEnv, target_roll_rate: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining a target roll angular velocity around x-axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll_vel = asset.data.root_ang_vel_b[:, 0]
    error = roll_vel - target_roll_rate
    return torch.exp(-(error**2) / (target_roll_rate * 0.65) ** 2)


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
