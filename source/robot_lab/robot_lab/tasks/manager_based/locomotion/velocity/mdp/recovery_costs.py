"""Tensor calculations for recovery effort, independent of the simulator."""

import torch


def recovery_gate(up, trunk_force, tilt_start=0.85, tilt_full=0.5, contact_start=5.0, contact_full=25.0):
    """Fade in costs below 32 degrees of upright or while the trunk bears load."""
    tilt = ((tilt_start - up) / (tilt_start - tilt_full)).clamp(0.0, 1.0)
    contact = ((trunk_force - contact_start) / (contact_full - contact_start)).clamp(0.0, 1.0)
    return torch.maximum(tilt, contact)


def normalized_effort(torque, effort_limit):
    return (torque / effort_limit.clamp_min(1.0e-6)).square().mean(dim=-1)


def normalized_velocity(velocity, reference_speed):
    return (velocity / reference_speed).square().mean(dim=-1)


def distance_energy_score(power, motion_speed, tracking, active, energy_per_metre=500.0, speed_floor=0.1):
    """Bounded absolute mechanical-work proxy; stationary commands receive no travel reward."""
    denominator = energy_per_metre * motion_speed.clamp_min(speed_floor)
    return torch.exp(-power / denominator) * tracking * active


class TorqueChangeHistory:
    """Track every control step, including upright steps; suppress reset discontinuities."""

    def __init__(self, initial_torque):
        self.previous = torch.zeros_like(initial_torque)
        self.valid = torch.zeros(initial_torque.shape[0], device=initial_torque.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.valid[:] = False
        else:
            self.valid[env_ids] = False

    def compute(self, torque, effort_limit, dt, reference_dt=0.02):
        delta = (torque - self.previous) * (reference_dt / dt)
        cost = normalized_effort(delta, effort_limit) * self.valid
        self.previous.copy_(torque)
        self.valid[:] = True
        return cost
