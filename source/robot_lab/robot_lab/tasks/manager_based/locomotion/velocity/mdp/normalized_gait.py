"""Reward gait coordination and cadence from measured, completed foot cycles."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def normalized_timing_score(
    air_time: torch.Tensor,
    contact_time: torch.Tensor,
    periods: torch.Tensor,
    phase_error_scale: float,
    max_phase_error: float,
) -> torch.Tensor:
    """Original two-sync/four-async score, using dimensionless timing errors.

    Columns 0/1 form one synchronized pair and 2/3 the other. Each comparison
    uses the mean of its two feet's completed periods. Inputs must have positive
    periods; the manager term separately masks incomplete/invalid histories.
    ``phase_error_scale`` divides squared errors, as ``std`` did in GaitReward.
    This is normalized elapsed-time mismatch, not circular oscillator phase.
    """
    first, second = [0, 2, 0, 1, 0, 2], [1, 3, 2, 3, 3, 1]
    pair_period = 0.5 * (periods[:, first] + periods[:, second])
    # First two columns compare like contact modes; last four compare opposite
    # modes. Sum exponents is equivalent to the original six-factor product.
    other_air = torch.cat((air_time[:, second[:2]], contact_time[:, second[2:]]), dim=1)
    other_contact = torch.cat((contact_time[:, second[:2]], air_time[:, second[2:]]), dim=1)
    air_error = (air_time[:, first] - other_air) / pair_period
    contact_error = (contact_time[:, first] - other_contact) / pair_period
    error = air_error.square().clamp(max=max_phase_error**2)
    error += contact_error.square().clamp(max=max_phase_error**2)
    return torch.exp(-error.sum(dim=1) / phase_error_scale)


class _CompletedFootCycleReward(ManagerTermBase):
    """Shared contact-event bookkeeping; subclasses choose the reward score."""

    def __init__(self, cfg, env, foot_names=None):
        super().__init__(cfg, env)
        params = cfg.params
        self.min_period, self.max_period = params["period_range"]
        if not (
            math.isfinite(self.min_period) and math.isfinite(self.max_period)
            and 2 * env.step_dt <= self.min_period < self.max_period
        ):
            raise ValueError("period_range must be finite, ordered, and span at least two control steps")
        self.stale_periods = params["stale_periods"]
        if not math.isfinite(self.stale_periods) or self.stale_periods < 1:
            raise ValueError("stale_periods must be finite and at least one full period")
        self.sensor = env.scene[params["sensor_cfg"].name]
        self.asset = env.scene[params["asset_cfg"].name]
        if not self.sensor.cfg.track_air_time:
            raise ValueError("Measured foot cycles require ContactSensor.track_air_time=True")
        selected = params["sensor_cfg"].body_ids
        if isinstance(selected, slice):
            selected = list(range(len(self.sensor.body_names)))[selected]
        by_name = {self.sensor.body_names[i]: i for i in selected}
        names = list(by_name) if foot_names is None else foot_names
        if not names or len(set(names)) != len(names):
            raise ValueError("At least one distinct contact sensor body must be selected")
        if any(name not in by_name for name in names):
            raise ValueError("A gait foot is absent from the selected contact sensor bodies")
        self.foot_ids = [by_name[name] for name in names]
        like = self.sensor.data.current_contact_time[:, self.foot_ids]
        # Unit placeholders are never used for valid rewards before a cycle
        # exists; they only keep masked intermediate divisions finite.
        self.periods = torch.ones_like(like)
        self.valid_periods = torch.zeros_like(like, dtype=torch.bool)
        self._landing_step = torch.full_like(like, -1, dtype=torch.long)
        self._landing_age = torch.zeros_like(like)
        self._last_processed_step = -1

    def reset(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        self.periods[indices] = 1.0
        self.valid_periods[indices] = False
        self._landing_step[indices] = -1
        self._landing_age[indices] = 0.0
        # A partial reset must not reopen event processing for other envs in
        # this same control step.

    def _update_periods(self, env, contact_time, last_air_time, last_contact_time, active):
        step = env.common_step_counter
        if step == self._last_processed_step:
            return
        self._last_processed_step = step
        self.reset(~active)
        # Sensor timestamps accumulate in float32. A few microseconds of slack
        # preserves exact control-step boundaries without admitting another
        # physics sample (5 ms in this task).
        tolerance = 1.0e-5
        touchdown = self.sensor.compute_first_contact(env.step_dt, abs_tol=tolerance)[:, self.foot_ids]
        # Initial contact after spawning is not a completed air-to-contact event.
        touchdown = touchdown & (last_air_time > 0) & active.unsqueeze(1)
        elapsed = (step - self._landing_step).to(contact_time.dtype) * env.step_dt + self._landing_age
        # A touchdown within this control step must be tested at its event time,
        # not at the later reward sample (which could incorrectly exceed 2*T).
        history_age = torch.where(touchdown, elapsed - contact_time, elapsed)
        stale = (self._landing_step >= 0) & (
            (self.valid_periods & (history_age > self.stale_periods * self.periods + tolerance))
            | (~self.valid_periods & (history_age > self.max_period + tolerance))
        )
        self.valid_periods[stale] = False
        self._landing_step[stale] = -1
        candidate = (step - self._landing_step).to(contact_time.dtype) * env.step_dt
        candidate = candidate - contact_time + self._landing_age
        # ContactSensor includes one physics update at the final transition in
        # last_air_time. Remove it only for checking resolvable phase lengths;
        # period estimation itself uses differences of touchdown timestamps.
        air_duration = last_air_time - env.physics_dt
        contact_duration = last_contact_time - env.physics_dt
        # A missed touchdown within a control step must not turn several real
        # cycles into one inflated denominator. Use the two completed phases
        # only to validate the timestamp-derived period, not to estimate it.
        complete_cycle = (candidate - air_duration - contact_duration).abs() <= env.physics_dt + tolerance
        valid = (
            touchdown & (self._landing_step >= 0) & torch.isfinite(candidate)
            & (candidate >= self.min_period - tolerance) & (candidate <= self.max_period + tolerance)
            & (air_duration >= env.step_dt - tolerance) & (contact_duration >= env.step_dt - tolerance)
            & complete_cycle
        )
        self.periods = torch.where(valid, candidate, self.periods)
        self.valid_periods = torch.where(touchdown, valid, self.valid_periods)
        self._landing_step = torch.where(touchdown, step, self._landing_step)
        self._landing_age = torch.where(touchdown, contact_time, self._landing_age)


class PeriodNormalizedGaitReward(_CompletedFootCycleReward):
    """Estimate each foot's last touchdown-to-touchdown period, then score.

    Startup, stopping, reset, rejected cycles and stale histories earn zero
    until all four feet have completed valid cycles. There is no fixed-period
    fallback, period clipping or target frequency. Only reward state changes;
    actor observations and deployed policy inputs are unaffected.
    """

    def __init__(self, cfg, env):
        params = cfg.params
        for key in ("phase_error_scale", "max_phase_error"):
            if not math.isfinite(params[key]) or params[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        pairs = params["synced_feet_pair_names"]
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError("Exactly two synchronized pairs are required")
        names = [name for pair in pairs for name in pair]
        if len(set(names)) != 4:
            raise ValueError("Gait feet must be four distinct body names")
        super().__init__(cfg, env, names)

    def __call__(
        self, env, phase_error_scale: float, max_phase_error: float,
        command_name: str, velocity_threshold: float, command_threshold: float,
        synced_feet_pair_names, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
        period_range, stale_periods: float,
    ) -> torch.Tensor:
        command = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
        speed = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        active = (command > command_threshold) | (speed > velocity_threshold)
        data = self.sensor.data
        air = data.current_air_time[:, self.foot_ids]
        contact = data.current_contact_time[:, self.foot_ids]
        last_air = data.last_air_time[:, self.foot_ids]
        last_contact = data.last_contact_time[:, self.foot_ids]
        self._update_periods(env, contact, last_air, last_contact, active)
        score = normalized_timing_score(air, contact, self.periods, phase_error_scale, max_phase_error)
        score = torch.where(active & self.valid_periods.all(dim=1), score, 0.0)
        return score * torch.clamp(-self.asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def cadence_score(
    periods: torch.Tensor,
    elapsed: torch.Tensor,
    target_frequency: float,
    frequency_scale: float,
    valid_periods: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bounded continuous score of actual Hz, including overdue open cycles.

    ``periods`` contains completed touchdown-to-touchdown intervals. During
    an unfinished cycle, its elapsed time is a lower bound on its eventual
    period. Once that bound exceeds the target period, it supplies a growing
    slow-cadence error. Taking the larger error prevents waiting from improving
    a previously fast cycle without first completing a new measured cycle.

    The rational kernel scores each foot independently, retaining useful
    shaping far from the target. An incomplete or stale foot earns zero, but
    does not erase other feet's measured cycles. The denominator is always the
    total selected foot count, so dropping a poor foot's valid history cannot
    increase the score. Omitting ``valid_periods`` treats all feet as valid.
    At most one unit score is available per control step, independent of how
    many touchdowns occurred.
    """
    completed_error = (periods.reciprocal() - target_frequency).abs()
    overdue_error = (target_frequency - elapsed.clamp_min(1.0e-8).reciprocal()).clamp_min(0.0)
    error = torch.maximum(completed_error, overdue_error) / frequency_scale
    per_foot_score = (1.0 + error.square()).reciprocal()
    if valid_periods is not None:
        per_foot_score = torch.where(valid_periods, per_foot_score, 0.0)
    return per_foot_score.mean(dim=1)


class FootCadenceReward(_CompletedFootCycleReward):
    """Track a specified full-cycle frequency while a motion command is active.

    A frequency of 2 Hz means each selected foot completes a touchdown-to-
    touchdown cycle every 0.5 s. This is neither aggregate touchdown rate nor
    inverse air time, and it does not prescribe the stance/swing duty factor.
    Each foot earns its own share once it has a valid completed cycle; missing
    or stale histories earn zero for that foot. Initial standing, reset, and
    zero commands earn zero. No actor observations or controller clock are
    added.
    """

    def __init__(self, cfg, env):
        params = cfg.params
        for key in ("target_frequency", "frequency_scale"):
            if not math.isfinite(params[key]) or params[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        super().__init__(cfg, env)
        if not self.min_period <= 1.0 / params["target_frequency"] <= self.max_period:
            raise ValueError("The target cadence period must be inside period_range")

    def __call__(
        self, env, target_frequency: float, frequency_scale: float,
        command_name: str, command_threshold: float,
        asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
        period_range, stale_periods: float,
    ) -> torch.Tensor:
        command = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
        active = command > command_threshold
        data = self.sensor.data
        contact = data.current_contact_time[:, self.foot_ids]
        last_air = data.last_air_time[:, self.foot_ids]
        last_contact = data.last_contact_time[:, self.foot_ids]
        self._update_periods(env, contact, last_air, last_contact, active)
        elapsed = (env.common_step_counter - self._landing_step).to(contact.dtype) * env.step_dt
        elapsed = elapsed + self._landing_age
        score = cadence_score(self.periods, elapsed, target_frequency, frequency_scale, self.valid_periods)
        score = torch.where(active, score, 0.0)
        return score * torch.clamp(-self.asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
