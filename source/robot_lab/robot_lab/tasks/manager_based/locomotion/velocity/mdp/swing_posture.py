"""Reduce the standing-pose prior for selected joints of commanded swing feet."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg


class SwingJointPositionPenalty(ManagerTermBase):
    """Keep the original posture norm, with weaker selected swing-joint errors.

    ``swing_scale`` multiplies joint deviations before the L2 norm. A value of
    0.1 retains 10% of a pure swing-joint penalty, not 10% of the entire term.
    Zero excludes the mapped swing joints from the norm entirely; the reward
    term's external weight independently scales the remaining penalty.
    Feet use the same current-air-time definition as the foot-height reward;
    this does not establish that the rest of the leg is free of contact.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        scale = params["swing_scale"]
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("swing_scale must be finite and in [0, 1]")
        asset_cfg, sensor_cfg = params["asset_cfg"], params["sensor_cfg"]
        self.robot = env.scene[asset_cfg.name]
        self.sensor = env.scene[sensor_cfg.name]
        if not self.sensor.cfg.track_air_time:
            raise ValueError("Swing posture requires ContactSensor.track_air_time=True")
        self.joint_ids = asset_cfg.joint_ids
        selected_ids = self.joint_ids
        if isinstance(selected_ids, slice):
            selected_ids = list(range(len(self.robot.joint_names)))[selected_ids]
        joint_names = [self.robot.joint_names[i] for i in selected_ids]
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Posture joints must be unique")
        sensor_ids = sensor_cfg.body_ids
        if isinstance(sensor_ids, slice):
            sensor_ids = list(range(len(self.sensor.body_names)))[sensor_ids]
        foot_ids = {self.sensor.body_names[i]: i for i in sensor_ids}
        columns, contacts = [], []
        for foot, joints in params["foot_joint_names"].items():
            if foot not in foot_ids:
                raise ValueError(f"Swing foot is absent from the selected contact bodies: {foot}")
            if not joints:
                raise ValueError(f"No swing joints configured for {foot}")
            for joint in joints:
                if joint not in joint_names:
                    raise ValueError(f"Swing joint is absent from the posture term: {joint}")
                columns.append(joint_names.index(joint))
                contacts.append(foot_ids[foot])
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Swing joints must be nonempty and assigned to exactly one foot")
        self.swing_columns = torch.tensor(columns, dtype=torch.long, device=env.device)
        self.foot_ids = torch.tensor(contacts, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        stand_still_scale: float,
        velocity_threshold: float,
        command_threshold: float,
        sensor_cfg: SceneEntityCfg,
        foot_joint_names: dict[str, tuple[str, ...]],
        swing_scale: float,
    ) -> torch.Tensor:
        data = self.robot.data
        command = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
        body_speed = torch.linalg.norm(data.root_lin_vel_b[:, :2], dim=1)
        error = data.joint_pos[:, self.joint_ids] - data.default_joint_pos[:, self.joint_ids]
        # Require a movement command, even if the robot is drifting while
        # commanded to stop. Keep the old speed gate only for standing scaling.
        swing = self.sensor.data.current_air_time[:, self.foot_ids] > 0.0
        relax = swing & (command > command_threshold).unsqueeze(1)
        error[:, self.swing_columns] *= torch.where(relax, error.new_tensor(swing_scale), 1.0)
        cost = torch.linalg.norm(error, dim=1)
        moving = (command > command_threshold) | (body_speed > velocity_threshold)
        cost = torch.where(moving, cost, stand_still_scale * cost)
        return cost * torch.clamp(-data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
