"""Reset helpers for the hip-only rolling task."""

import torch

from isaaclab.managers import SceneEntityCfg


def reset_hips_to_default(env, env_ids, asset_cfg: SceneEntityCfg):
    """Start at the exact support pose, without soft-limit clipping or velocity noise."""
    asset = env.scene[asset_cfg.name]
    positions = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    limits = asset.data.joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    positions.clamp_(limits[..., 0], limits[..., 1])
    velocities = torch.zeros_like(positions)
    asset.write_joint_state_to_sim(positions, velocities, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(positions, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target(velocities, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
