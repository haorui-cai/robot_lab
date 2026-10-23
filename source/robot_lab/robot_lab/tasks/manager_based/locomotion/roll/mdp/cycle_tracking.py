"""Tensor-only measurements of completed, signed rolling revolutions."""

import math

import torch


class CycleWindow:
    """Dense average over one commanded period, with fractional-step interpolation."""

    def __init__(self, count, device, capacity=128):
        self.history = torch.zeros(count, capacity, 2, device=device)
        self.integral = torch.zeros(count, 2, device=device)
        self.age = torch.zeros(count, device=device)
        self.mean = torch.zeros(count, 2, device=device)
        self.cursor = 0

    def reset(self, ids):
        for value in (self.history, self.integral, self.age, self.mean):
            value[ids] = 0

    def update(self, rates, period, dt):
        self.integral += rates * dt
        self.age += dt
        self.history[:, self.cursor] = self.integral
        steps = (period / dt).clamp_max(self.history.shape[1] - 2)
        whole = steps.long()
        rows = torch.arange(len(rates), device=rates.device)
        recent = (self.cursor - whole) % self.history.shape[1]
        older = (recent - 1) % self.history.shape[1]
        past = torch.lerp(self.history[rows, recent], self.history[rows, older], (steps - whole).unsqueeze(-1))
        past = torch.where((self.age < period).unsqueeze(-1), 0.0, past)
        duration = torch.minimum(self.age, period).clamp_min(dt)
        self.mean[:] = (self.integral - past) / duration.unsqueeze(-1)
        self.cursor = (self.cursor + 1) % self.history.shape[1]


class ActualCycleWindow:
    """Trailing signed negative revolution, integrated in time, not angle-weighted.

    Find the latest past crossing of current signed progress minus 2*pi.
    Reversals subtract progress; dwelling still increases elapsed time. No full
    revolution means invalid credit, rather than reusing a stale completed lap.
    The caller reserves the entire episode so a slow cycle is never truncated.
    """

    def __init__(self, count, device, capacity=1024):
        if capacity < 2:
            raise ValueError("Actual-cycle history requires at least two samples")
        # Double precision avoids cancellation when subtracting cumulative times
        # and angles, particularly at the strict minimum-speed reward gate.
        self.history = torch.zeros(count, capacity, 3, device=device, dtype=torch.float64)
        self.history[:, :, 1] = -1  # invalid time before each environment's reset
        self.current = torch.zeros(count, 3, device=device, dtype=torch.float64)
        self.mean = torch.zeros(count, 2, device=device)
        self.duration = torch.zeros(count, device=device)
        self.valid = torch.zeros(count, device=device, dtype=torch.bool)
        self.cursor = 0

    def reset(self, ids):
        self.history[ids] = 0
        self.history[ids, :, 1] = -1
        for value in (self.current, self.mean, self.duration, self.valid):
            value[ids] = 0

    def update(self, roll_rate, yaw_rate, dt):
        self.history[:, self.cursor] = self.current
        self.current[:, 0] -= roll_rate.to(torch.float64) * dt
        self.current[:, 1] += dt
        self.current[:, 2] += yaw_rate.to(torch.float64) * dt
        target = self.current[:, 0] - 2 * math.pi
        times = self.history[:, :, 1]
        eligible = (times >= 0) & (self.history[:, :, 0] <= target[:, None])
        lower_time, index = torch.where(eligible, times, -1.0).max(dim=1)
        rows = torch.arange(len(roll_rate), device=roll_rate.device)
        lower = self.history[rows, index]
        upper = self.history[rows, (index + 1) % self.history.shape[1]]
        upper = torch.where((index == self.cursor)[:, None], self.current, upper)
        fraction = ((target - lower[:, 0]) / (upper[:, 0] - lower[:, 0]).clamp_min(1e-12)).clamp(0, 1)
        start = torch.lerp(lower, upper, fraction[:, None])
        duration = (self.current[:, 1] - start[:, 1]).clamp_min(1e-12)
        self.valid[:] = lower_time >= 0
        self.duration[:] = torch.where(self.valid, duration, 0.0)
        means = torch.stack((-2 * math.pi / duration, (self.current[:, 2] - start[:, 2]) / duration), -1)
        self.mean[:] = torch.where(self.valid[:, None], means, 0.0)
        self.cursor = (self.cursor + 1) % self.history.shape[1]


class CycleTracker:
    """Track curriculum revolutions and configurable rolling reward averages.

    Legacy tasks use the commanded-period window. Left-thigh tasks opt into a
    trailing actual revolution for rewards while preserving policy observations.
    """

    def __init__(self, count, device, actual_cycle_rewards=False, history_capacity=1024, yaw_min_speed=None,
                 yaw_tracking_std=0.25, yaw_success_tolerance=0.15, legacy_reward_fraction=0.0):
        if not math.isfinite(yaw_tracking_std) or yaw_tracking_std <= 0:
            raise ValueError("Yaw tracking scale must be finite and positive")
        if not math.isfinite(yaw_success_tolerance) or yaw_success_tolerance < 0:
            raise ValueError("Yaw success tolerance must be finite and nonnegative")
        if not math.isfinite(legacy_reward_fraction) or not 0 <= legacy_reward_fraction <= 1:
            raise ValueError("Legacy reward fraction must be within [0, 1]")
        if legacy_reward_fraction and not actual_cycle_rewards:
            raise ValueError("Blending historical rewards requires actual-cycle rewards")
        self.yaw_tracking_std = yaw_tracking_std
        self.yaw_success_tolerance = yaw_success_tolerance
        self.legacy_reward_fraction = legacy_reward_fraction
        for name in (
            "progress", "elapsed", "yaw_integral", "roll_payment", "yaw_payment",
            "cycles", "roll_good", "yaw_good", "last_roll", "last_yaw",
            "previous_roll", "roll_acceleration", "previous_yaw", "yaw_acceleration",
        ):
            setattr(self, name, torch.zeros(count, device=device))
        self.window = CycleWindow(count, device)
        # Keep the legacy observation window unchanged for checkpoint/deployment
        # compatibility; only the explicitly enabled reward window changes.
        self.actual_window = ActualCycleWindow(count, device, history_capacity) if actual_cycle_rewards else None
        self.yaw_min_speed = yaw_min_speed
        self.previous_roll_valid = torch.zeros(count, device=device, dtype=torch.bool)
        self.previous_yaw_valid = torch.zeros(count, device=device, dtype=torch.bool)

    def reset(self, ids, initial_roll_rate=None, initial_yaw_rate=None):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
        self.window.reset(ids)
        if self.actual_window is not None:
            self.actual_window.reset(ids)
        if initial_roll_rate is not None:
            self.previous_roll[ids] = initial_roll_rate
            self.previous_roll_valid[ids] = True
        if initial_yaw_rate is not None:
            self.previous_yaw[ids] = initial_yaw_rate
            self.previous_yaw_valid[ids] = True

    def update(self, roll_rate, yaw_rate, command, dt):
        self.roll_acceleration[:] = torch.where(
            self.previous_roll_valid, (roll_rate - self.previous_roll) / dt, 0.0
        )
        self.previous_roll[:] = roll_rate
        self.previous_roll_valid[:] = True
        self.yaw_acceleration[:] = torch.where(
            self.previous_yaw_valid, (yaw_rate - self.previous_yaw) / dt, 0.0
        )
        self.previous_yaw[:] = yaw_rate
        self.previous_yaw_valid[:] = True
        period = 2 * math.pi / command[:, 0].abs().clamp_min(1)
        self.window.update(torch.stack((roll_rate, yaw_rate), dim=-1), period, dt)
        reward_window = self.window
        if self.actual_window is not None:
            self.actual_window.update(roll_rate, yaw_rate, dt)
            reward_window = self.actual_window
        self.roll_payment[:] = torch.exp(-((reward_window.mean[:, 0] - command[:, 0]) / 1.5).square())
        if self.actual_window is not None:
            self.roll_payment *= self.actual_window.valid
        yaw_score = torch.exp(-((reward_window.mean[:, 1] - command[:, 1]) / self.yaw_tracking_std).square())
        if self.yaw_min_speed is None:
            self.yaw_payment[:] = self.roll_payment * yaw_score
        else:
            # Commands are negative. Numerical tolerance keeps exact -6 at the
            # closed gate; magnitude alone would incorrectly accept reversal.
            gate = -reward_window.mean[:, 0] > self.yaw_min_speed + 1e-6
            if self.actual_window is not None:
                gate &= self.actual_window.valid
            self.yaw_payment[:] = gate * yaw_score
        if self.legacy_reward_fraction:
            # Average the two historical reward outputs, not their scales or
            # windows. The older branch intentionally pays before a full lap.
            legacy_roll = torch.exp(-((self.window.mean[:, 0] - command[:, 0]) / 1.5).square())
            legacy_yaw = legacy_roll * torch.exp(-((self.window.mean[:, 1] - command[:, 1]) / 0.25).square())
            self.roll_payment.lerp_(legacy_roll, self.legacy_reward_fraction)
            self.yaw_payment.lerp_(legacy_yaw, self.legacy_reward_fraction)
        travel = -roll_rate * dt
        complete = self.progress + travel >= 2 * math.pi
        fraction = ((2 * math.pi - self.progress) / travel.clamp_min(1e-8)).clamp(0, 1)
        duration = self.elapsed + fraction * dt
        roll_avg = -2 * math.pi / duration.clamp_min(dt * 0.01)
        yaw_avg = (self.yaw_integral + fraction * dt * yaw_rate) / duration.clamp_min(dt * 0.01)
        roll_error = roll_avg - command[:, 0]
        yaw_error = yaw_avg - command[:, 1]
        self.cycles += complete
        self.roll_good += complete & (roll_error.abs() <= 0.20 * command[:, 0].abs())
        # Float32 integration can put an exact boundary a few ulps above it.
        self.yaw_good += complete & (yaw_error.abs() <= self.yaw_success_tolerance + 1e-7)
        self.last_roll[complete] = roll_avg[complete]
        self.last_yaw[complete] = yaw_avg[complete]
        self.progress += travel
        self.elapsed += dt
        self.yaw_integral += yaw_rate * dt
        self.progress[complete] -= 2 * math.pi
        self.elapsed[complete] = ((1 - fraction) * dt)[complete]
        self.yaw_integral[complete] = ((1 - fraction) * dt * yaw_rate)[complete]


class RollingCourse:
    """Advance only after two sufficiently large successful evaluation batches."""

    phase_widths = (0.10, 0.20, 0.35, 0.55, 0.80, 1.20, 1.80, math.pi)
    yaw_widths = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)

    def __init__(self, initial_speed_scales=(1.0,)):
        if not initial_speed_scales or initial_speed_scales[0] != 1.0 or any(
            not 0 <= value <= 1 for value in initial_speed_scales
        ) or any(a <= b for a, b in zip(initial_speed_scales, initial_speed_scales[1:])):
            raise ValueError("Initial speed scales must start at 1 and strictly decrease within [0, 1]")
        self.initial_speed_scales = tuple(initial_speed_scales)
        self.assist_level = 0
        self.phase_level = 0
        self.yaw_level = 0
        self.count = self.roll_success = self.joint_success = 0
        self.streak = 0
        self.last_step = 0
        self.axes = ("assist", "phase", "yaw") if len(initial_speed_scales) > 1 else ("phase", "yaw")
        self.next_axis = self.axes[0]

    @property
    def initial_speed_scale(self):
        return self.initial_speed_scales[self.assist_level]

    def observe(self, count, roll_success, joint_success, step, batch_size=2048, min_steps=500):
        self.count += count
        self.roll_success += roll_success
        self.joint_success += joint_success
        if self.count < batch_size or step - self.last_step < min_steps:
            return False
        # Require both tasks to remain successful as either difficulty expands.
        passed = self.roll_success / self.count >= 0.75 and self.joint_success / self.count >= 0.70
        self.streak = self.streak + 1 if passed else 0
        self.count = self.roll_success = self.joint_success = 0
        self.last_step = step
        if self.streak < 2:
            return False
        self.streak = 0
        limits = {"phase": len(self.phase_widths), "yaw": len(self.yaw_widths),
                  "assist": len(self.initial_speed_scales)}
        start = self.axes.index(self.next_axis)
        for offset in range(len(self.axes)):
            index = (start + offset) % len(self.axes)
            axis = self.axes[index]
            level = getattr(self, f"{axis}_level")
            if level < limits[axis] - 1:
                setattr(self, f"{axis}_level", level + 1)
                self.next_axis = self.axes[(index + 1) % len(self.axes)]
                return True
        return False
