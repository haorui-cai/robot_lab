"""Acceleration regularizers at the corrected virtual base, in world axes."""

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .base_kinematics import AccelerationHistory, TQBOT2_BASE_OFFSET, base_point_velocity


class BaseAccelerationL2(ManagerTermBase):
    """Control-step derivative of base-point velocity or rigid-body angular velocity.

    World Z stays vertical during rolling. This is kinematic acceleration (free
    fall includes gravity), not IMU specific force. The first sample after each
    reset establishes history and contributes zero. Angular acceleration is the
    same at base and trunk because they are fixed on the same rigid body.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        if self.robot.body_names[0] != "trunk":
            raise ValueError("TQBot2 base offset requires the physical trunk as root")
        self.history = AccelerationHistory(env.num_envs, env.device)
        self.offset = torch.tensor(cfg.params["offset_b"], device=env.device).expand(env.num_envs, -1)
        self.axis_weights = torch.tensor(cfg.params["axis_weights"], device=env.device)

    def reset(self, env_ids=None):
        self.history.reset(env_ids)

    def __call__(self, env, angular=False, axis_weights=(1.0, 1.0, 2.0), offset_b=TQBOT2_BASE_OFFSET):
        data = self.robot.data
        velocity = data.root_ang_vel_w
        if not angular:
            velocity = base_point_velocity(
                data.root_link_lin_vel_w, velocity, data.root_link_quat_w, self.offset
            )
        acceleration = self.history.update(velocity, env.step_dt, env.common_step_counter)
        return (acceleration.square() * self.axis_weights).sum(-1)


def base_acceleration_rewards():
    """Fresh configs shared by legacy, hip-only, left-thigh and slope rolling."""
    return {
        "base_linear_acceleration": RewardTermCfg(
            func=BaseAccelerationL2, weight=-1.0e-4,
            params={"angular": False, "axis_weights": (1.0, 1.0, 2.0), "offset_b": TQBOT2_BASE_OFFSET},
        ),
        "base_angular_acceleration": RewardTermCfg(
            func=BaseAccelerationL2, weight=-2.0e-6,
            params={"angular": True, "axis_weights": (1.0, 1.0, 2.0), "offset_b": TQBOT2_BASE_OFFSET},
        ),
    }
