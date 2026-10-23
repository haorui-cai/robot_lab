"""World-frame derivatives at the fitted TQBot2 virtual base (torch only)."""

import torch

# In physical trunk coordinates; unitree_guide commit 0211114, floating_base.
TQBOT2_BASE_OFFSET = (0.0, 0.0000611544755962, -0.0201840868229783)


def base_point_velocity(link_velocity_w, angular_velocity_w, quaternion_w, offset_b):
    """Transport link-origin velocity to base, including omega cross r.

    quaternion_w is wxyz. The input must be LINK-origin velocity, not COM velocity.
    """
    q_vec = quaternion_w[:, 1:]
    cross = 2.0 * torch.linalg.cross(q_vec, offset_b, dim=-1)
    offset_w = offset_b + quaternion_w[:, :1] * cross + torch.linalg.cross(q_vec, cross, dim=-1)
    return link_velocity_w + torch.linalg.cross(angular_velocity_w, offset_w, dim=-1)


class AccelerationHistory:
    """Finite difference in a fixed world frame; reset and repeated reads are safe."""

    def __init__(self, num_envs, device):
        self.previous = torch.zeros(num_envs, 3, device=device)
        self.acceleration = torch.zeros_like(self.previous)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.step = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.valid[ids] = False
        self.acceleration[ids] = 0.0

    def update(self, velocity_w, dt, step):
        if step != self.step:
            self.acceleration[:] = torch.where(
                self.valid[:, None], (velocity_w - self.previous) / dt, 0.0
            )
            self.previous.copy_(velocity_w)
            self.valid[:] = True
            self.step = step
        return self.acceleration
