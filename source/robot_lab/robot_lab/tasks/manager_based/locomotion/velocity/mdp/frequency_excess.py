"""One-sided cadence cost from debounced contacts at physics-step resolution."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def frequency_excess_cost(periods, measured, threshold_frequency, frequency_scale):
    """Mean of four independent costs, quadratic near the limit, linear beyond.

    Missing cycles contribute zero without changing the denominator. The term
    maintains the last measured period separately, so a brief contact glitch
    or a long wait cannot erase an already measured excess.
    """
    safe_periods = torch.where(measured, periods, 1.0)
    excess = ((safe_periods.reciprocal() - threshold_frequency) / frequency_scale).clamp_min(0.0)
    cost = torch.where(excess <= 1.0, excess.square(), 2.0 * excess - 1.0)
    return torch.where(measured, cost, 0.0).mean(dim=1)


class FootFrequencyExcess(ManagerTermBase):
    """Penalize fast completed cycles, including those below 80 ms.

    Read every physics sample from a dedicated foot contact sensor. A state
    transition requires consecutive samples (10 ms by default) and is dated
    to the first confirming sample. Initial contact is not a touchdown. Two
    air-to-contact events are needed before a foot has a measured period.

    The last cost persists until a new completed cycle, a stop command, or an
    episode reset. There is no timeout or fabricated frequency for a foot that
    has never stepped. Existing positive cadence and tracking rewards address
    inactivity. This term changes no observations or controller inputs.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        for name in ("threshold_frequency", "frequency_scale", "confirmation_time", "force_threshold"):
            if not math.isfinite(params[name]) or params[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(params["command_threshold"]) or params["command_threshold"] < 0:
            raise ValueError("command_threshold must be finite and nonnegative")
        self.physics_dt = env.physics_dt
        self.samples_per_step = round(env.step_dt / self.physics_dt)
        if self.samples_per_step < 1 or not math.isclose(
            env.step_dt, self.samples_per_step * self.physics_dt, abs_tol=1.e-9
        ):
            raise ValueError("A control step must contain an integer number of physics samples")
        self.confirm_samples = round(params["confirmation_time"] / self.physics_dt)
        if self.confirm_samples < 2 or not math.isclose(
            params["confirmation_time"], self.confirm_samples * self.physics_dt, abs_tol=1.e-9
        ):
            raise ValueError("confirmation_time must span at least two whole physics samples")
        self.sensor = env.scene[params["sensor_cfg"].name]
        self.asset = env.scene[params["asset_cfg"].name]
        if not math.isclose(self.sensor.cfg.update_period, self.physics_dt, abs_tol=1.e-9):
            raise ValueError("Foot frequency sensor must update every physics step")
        if self.sensor.cfg.history_length < self.samples_per_step:
            raise ValueError("Foot frequency sensor history must cover every physics sample in a control step")
        selected = params["sensor_cfg"].body_ids
        if isinstance(selected, slice):
            selected = list(range(len(self.sensor.body_names)))[selected]
        self.foot_ids = list(selected)
        if len(self.foot_ids) != 4 or len(set(self.foot_ids)) != 4:
            raise ValueError("Foot frequency excess requires exactly four distinct feet")
        like = self.sensor.data.net_forces_w_history[:, 0, self.foot_ids, 0]
        self.periods = torch.ones_like(like)
        self.measured = torch.zeros_like(like, dtype=torch.bool)
        self.completed_cycles = torch.zeros_like(like, dtype=torch.long)
        self._known = torch.zeros_like(self.measured)
        self._contact = torch.zeros_like(self.measured)
        self._raw_contact = torch.zeros_like(self.measured)
        self._run_length = torch.zeros_like(self.completed_cycles)
        self._landing_tick = torch.full_like(self.completed_cycles, -1)
        self._last_processed_step = env.common_step_counter

    def reset(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        self.periods[indices] = 1.0
        self.measured[indices] = False
        self.completed_cycles[indices] = 0
        self._known[indices] = False
        self._contact[indices] = False
        self._raw_contact[indices] = False
        self._run_length[indices] = 0
        self._landing_tick[indices] = -1
        # Do not reopen processing of other environments during partial reset.

    def _update(self, env, active, force_threshold):
        self.reset(~active)
        step = env.common_step_counter
        if step == self._last_processed_step:
            return
        if step != self._last_processed_step + 1:
            raise RuntimeError("Foot frequency history skipped a control step; cannot reconstruct complete cycles")
        history = self.sensor.data.net_forces_w_history[:, :self.samples_per_step, self.foot_ids, :]
        if not torch.isfinite(history).all():
            raise RuntimeError("Non-finite foot contact force; cadence cannot be measured")
        contacts = torch.linalg.vector_norm(history, dim=-1) > force_threshold
        enabled = active.unsqueeze(1)
        # Isaac Lab stores newest first. Read oldest first, including transitions
        # that begin in the final substep and are confirmed in the next step.
        for offset in range(self.samples_per_step):
            raw = contacts[:, self.samples_per_step - 1 - offset]
            length = torch.where(raw == self._raw_contact, self._run_length + 1, 1)
            self._run_length = torch.where(enabled, length.clamp(max=self.confirm_samples), 0)
            self._raw_contact = raw & enabled
            transition = (self._run_length >= self.confirm_samples) & (~self._known | (raw != self._contact))
            touchdown = transition & self._known & raw & enabled
            tick = (step - 1) * self.samples_per_step + offset + 1
            event_tick = tick - self.confirm_samples + 1
            complete = touchdown & (self._landing_tick >= 0)
            candidate = (event_tick - self._landing_tick).to(self.periods.dtype) * self.physics_dt
            self.periods = torch.where(complete, candidate, self.periods)
            self.measured |= complete
            self.completed_cycles += complete.long()
            self._landing_tick = torch.where(touchdown, event_tick, self._landing_tick)
            self._contact = torch.where(transition, raw, self._contact)
            self._known |= transition
        self._last_processed_step = step

    def __call__(
        self, env, threshold_frequency: float, frequency_scale: float,
        command_name: str, command_threshold: float, confirmation_time: float,
        force_threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        active = torch.linalg.vector_norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
        self._update(env, active, force_threshold)
        cost = frequency_excess_cost(self.periods, self.measured, threshold_frequency, frequency_scale)
        upright = torch.clamp(-self.asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
        return torch.where(active, cost, 0.0) * upright
