# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Recover body orientation on the ground; joint posture is unconstrained."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, random_orientation


def base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Height above the local flat ground (also used by the privileged critic)."""
    return (env.scene[asset_cfg.name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def reset_supine(env, env_ids: torch.Tensor):
    """Back on the floor, belly/feet upward: a 180-degree roll, never a drop.

    Random yaw changes heading only. The 0.096 m trunk height clears the
    imported shell's +Z extent (0.0948133 m) by about 1 mm when inverted.
    """
    robot = env.scene["robot"]
    count = len(env_ids)
    yaw = torch.empty(count, device=env.device).uniform_(-math.pi, math.pi)
    roll = torch.full_like(yaw, math.pi)
    pitch = torch.zeros_like(yaw)
    pose = torch.zeros(count, 7, device=env.device)
    pose[:, :3] = env.scene.env_origins[env_ids]
    pose[:, 2] += 0.096
    pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    joints = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
    robot.set_joint_position_target(joints, env_ids=env_ids)
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(count, 6, device=env.device), env_ids=env_ids)
    # Enforce the task contract at every full or partial reset, not just in tests.
    if not (robot.data.projected_gravity_b[env_ids, 2] > 0.99).all():
        raise RuntimeError("Recover reset must start inverted (back down, feet up)")


def reset_recovery(
    env,
    env_ids: torch.Tensor,
    easy_fraction: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Mix near-prone starts with uniform SO(3) orientations and full joint ranges.

    Start above the ground to avoid injecting large impulses from terrain
    penetration. The first second is reserved for landing before success counts.
    No orientation is excluded, including supine and head/tail-down poses.
    """
    robot = env.scene[asset_cfg.name]
    count = len(env_ids)
    easy = torch.rand(count, device=env.device) < easy_fraction
    limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = limits[..., 0] + torch.rand_like(limits[..., 0]) * (limits[..., 1] - limits[..., 0])
    near_pose = robot.data.default_joint_pos[env_ids] + torch.empty_like(joint_pos).uniform_(-0.15, 0.15)
    joint_pos = torch.where(easy[:, None], near_pose, joint_pos)
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    joint_vel = torch.empty_like(joint_pos).uniform_(-0.5, 0.5)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # Avoid a one-step kick toward the default pose before the first action.
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, :2] += torch.empty(count, 2, device=env.device).uniform_(-0.25, 0.25)
    # 0.75 m clears the trunk + fully extended legs at any root orientation.
    root[:, 2] += torch.where(easy, 0.30, 0.75)
    root[:, 3:7] = random_orientation(count, env.device)
    roll_pitch = torch.empty(count, 2, device=env.device).uniform_(-0.25, 0.25)
    yaw = torch.empty(count, device=env.device).uniform_(-math.pi, math.pi)
    easy_quat = quat_from_euler_xyz(roll_pitch[:, 0], roll_pitch[:, 1], yaw)
    root[:, 3:7] = torch.where(easy[:, None], easy_quat, root[:, 3:7])
    root[:, 7:10] = torch.empty(count, 3, device=env.device).uniform_(-0.2, 0.2)
    root[:, 10:13] = torch.empty(count, 3, device=env.device).uniform_(-0.5, 0.5)
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)


def upright(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Signed gravity distinguishes upright from upside down; yaw is free."""
    return ((1.0 - env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]) * 0.5).clamp(0.0, 1.0)


def prone_settled(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reward upright, low and quiet body; no preferred joint angles or speeds."""
    robot = env.scene[asset_cfg.name]
    # A low body is sufficient; do not track an exact FixedLie height.
    height_error = (base_height(env, asset_cfg).squeeze(-1) - 0.18).clamp_min(0.0).square()
    motion = robot.data.root_lin_vel_b.square().sum(-1) / 0.25**2
    motion += robot.data.root_ang_vel_b.square().sum(-1) / 0.75**2
    return upright(env, asset_cfg).square() * torch.exp(-height_error / 0.08**2 - motion)


def is_prone(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
):
    """A supported, low and upright body, regardless of joint configuration."""
    robot = env.scene[asset_cfg.name]
    height = base_height(env, asset_cfg).squeeze(-1)
    support = env.scene[sensor_cfg.name].data.net_forces_w.norm(dim=-1).amax(dim=-1) > 1.0
    return (
        (robot.data.projected_gravity_b[:, 2] < -0.95)
        & (height > 0.02)
        & (height < 0.18)
        & (robot.data.root_lin_vel_b.norm(dim=-1) < 0.15)
        & (robot.data.root_ang_vel_b.norm(dim=-1) < 0.5)
        & support
    )


class StableProne(ManagerTermBase):
    """End an episode only after uninterrupted prone stability, never in flight."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        self._stable_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, hold_time: float = 0.5, settle_time: float = 1.0):
        valid = is_prone(env) & (env.episode_length_buf * env.step_dt >= settle_time)
        self._stable_steps[:] = torch.where(valid, self._stable_steps + 1, 0)
        return self._stable_steps >= math.ceil(hold_time / env.step_dt)


def escaped(env, max_distance: float = 3.0):
    pos = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    return (pos[:, :2].norm(dim=-1) > max_distance) | (pos[:, 2] < -0.2) | (pos[:, 2] > 2.0)
