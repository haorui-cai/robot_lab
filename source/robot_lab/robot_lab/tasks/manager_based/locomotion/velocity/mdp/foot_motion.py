"""Tensor operations for a rigid foot point and ground-relative motion."""

import torch


def foot_point_state(position, quaternion, linear_velocity, angular_velocity, offset):
    """Inputs are LINK-origin states, quaternion wxyz; velocities are in world axes."""
    offset = torch.as_tensor(offset, device=position.device, dtype=position.dtype).expand_as(position)
    qvec = quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(qvec, offset, dim=-1)
    rotated = offset + quaternion[..., :1] * twice_cross + torch.cross(qvec, twice_cross, dim=-1)
    return position + rotated, linear_velocity + torch.cross(angular_velocity, rotated, dim=-1)


def tangent_velocity(velocity, normal):
    return velocity - (velocity * normal).sum(dim=-1, keepdim=True) * normal


def foot_motion_costs(clearance, foot_velocity, root_velocity, normal, contact, valid,
                      target_clearance=0.03, contact_clearance=0.02, motion_scale=0.2):
    """Stationary loaded feet cost zero; moving swing feet need a minimum clearance.

    Slip is measured against the stationary terrain. Body-relative motion is used
    only to identify a moving swing foot, never as a substitute for ground slip.
    """
    loaded = contact & (clearance < contact_clearance) & valid
    slip = tangent_velocity(foot_velocity, normal).norm(dim=-1) * loaded
    swing_motion = tangent_velocity(foot_velocity - root_velocity[:, None], normal).norm(dim=-1)
    deficit = ((target_clearance - clearance) / target_clearance).clamp(0.0, 1.0)
    swing = deficit.square() * (swing_motion / motion_scale).clamp(0.0, 1.0) * (~loaded & valid)
    return slip.mean(dim=1), swing.mean(dim=1)
