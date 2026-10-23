"""CPU contract tests for cycle-normalized gait; no Isaac Sim initialization."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp"
SOURCE = MDP / "normalized_gait.py"
CONFIG = MDP.parent / "config/quadruped/tqbot2/standard_walk_env_cfg.py"
PAIRS = (("FR_foot", "RL_foot"), ("FL_foot", "RR_foot"))
FEET = tuple(name for pair in PAIRS for name in pair)
DTYPE = torch.float64


class ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


def production_symbols():
    tree = ast.parse(SOURCE.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *definitions,
    ], type_ignores=[])
    namespace = {"torch": torch, "math": math, "ManagerTermBase": ManagerTermBase}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["normalized_timing_score"], namespace["PeriodNormalizedGaitReward"]


score, Reward = production_symbols()


def tensor(values):
    return torch.tensor(values, dtype=DTYPE)


class ContactSensor:
    """Use the installed ContactSensor's sampled timer transition convention."""

    def __init__(self, batch, body_names):
        self.body_names = body_names
        self.cfg = NS(track_air_time=True)
        self.timestamp = None
        self.data = NS(**{
            name: torch.zeros(batch, len(body_names), dtype=DTYPE)
            for name in ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time")
        })

    def physics_sample(self, contact, dt):
        # Enable the real float32 timestamp subtraction convention for the
        # boundary regression below; most cases use exact fixture intervals.
        if self.timestamp is not None:
            previous = self.timestamp.clone()
            self.timestamp += dt
            dt = (self.timestamp - previous).unsqueeze(-1)
        data = self.data
        first_contact = (data.current_air_time > 0) & contact
        first_air = (data.current_contact_time > 0) & ~contact
        data.last_air_time = torch.where(first_contact, data.current_air_time + dt, data.last_air_time)
        data.last_contact_time = torch.where(first_air, data.current_contact_time + dt, data.last_contact_time)
        data.current_air_time = torch.where(~contact, data.current_air_time + dt, 0.)
        data.current_contact_time = torch.where(contact, data.current_contact_time + dt, 0.)

    def compute_first_contact(self, dt, abs_tol=1e-8):
        return (self.data.current_contact_time > 0) & (self.data.current_contact_time < dt + abs_tol)

    def reset(self, rows=slice(None)):
        for value in vars(self.data).values():
            value[rows] = 0.
        if self.timestamp is not None:
            self.timestamp[rows] = 0.


class Fixture:
    def __init__(self, batch=1, body_names=None, params=None):
        self.sensor = ContactSensor(batch, list(body_names or ("trunk", *FEET)))
        self.asset = NS(data=NS(
            root_com_lin_vel_b=torch.zeros(batch, 3, dtype=DTYPE),
            projected_gravity_b=tensor([[0., 0., -1.]]).repeat(batch, 1),
        ))
        self.command = tensor([[.5, 0., 0.]]).repeat(batch, 1)
        self.env = NS(
            num_envs=batch, device="cpu", common_step_counter=0, step_dt=.02, physics_dt=.005,
            scene={"robot": self.asset, "contact_forces": self.sensor},
            command_manager=NS(get_command=lambda _: self.command),
        )
        self.params = dict(
            phase_error_scale=math.sqrt(.5), max_phase_error=.2,
            command_name="base_velocity", velocity_threshold=.5, command_threshold=.1,
            synced_feet_pair_names=PAIRS, asset_cfg=NS(name="robot"),
            sensor_cfg=NS(name="contact_forces", body_ids=slice(None)),
            period_range=(.08, 2.), stale_periods=2.,
        )
        self.params.update(params or {})
        self.term = Reward(NS(params=self.params), self.env)

    def evaluate(self):
        return self.term(self.env, **self.params)

    def contacts(self, feet):
        values = torch.as_tensor(feet, dtype=torch.bool)
        if values.ndim == 0:
            values = values.repeat(self.env.num_envs, 4)
        elif values.ndim == 1:
            values = values.unsqueeze(0).repeat(self.env.num_envs, 1)
        contact = torch.zeros_like(self.sensor.data.current_contact_time, dtype=torch.bool)
        for index, name in enumerate(FEET):
            contact[:, self.sensor.body_names.index(name)] = values[:, index]
        return contact

    def physics_pattern(self, pattern):
        """A sequence of foot contact states, length divisible by decimation."""
        assert len(pattern) % 4 == 0
        result = None
        for index, feet in enumerate(pattern):
            self.sensor.physics_sample(self.contacts(feet), self.env.physics_dt)
            if index % 4 == 3:
                self.env.common_step_counter += 1
                result = self.evaluate()
        return result

    def advance(self, feet, steps=1):
        return self.physics_pattern([feet] * (steps * 4))

    def prime(self, half_steps=5):
        """Observe two real landings separated by one full known cycle."""
        self.advance(True, half_steps)
        self.advance(False, half_steps)
        self.advance(True, half_steps)
        self.advance(False, half_steps)
        return self.advance(True, half_steps)


class NormalizedTimingScoreTests(unittest.TestCase):
    def evaluate(self, air, contact, periods):
        return score(tensor(air), tensor(contact), tensor(periods), math.sqrt(.5), .2)

    def test_same_fractional_mismatch_has_same_score(self):
        # 0.1/0.4 and 0.05/0.2 are both a quarter cycle. The three affected
        # pair comparisons each contribute the clipped 0.2**2 error.
        actual = self.evaluate([[.1, 0, 0, 0], [.05, 0, 0, 0]], [[0]*4]*2, [[.4]*4, [.2]*4])
        expected = math.exp(-3 * .2**2 / math.sqrt(.5))
        torch.testing.assert_close(actual, tensor([expected, expected]))

    def test_all_time_quantities_can_be_rescaled_without_changing_reward(self):
        air = tensor([[.03, .05, 0, 0], [.1, 0, 0, .03]])
        contact = tensor([[0, 0, .07, .09], [0, .12, .08, 0]])
        periods = tensor([[.2, .3, .4, .5], [.8, .6, .4, .2]])
        expected = score(air, contact, periods, math.sqrt(.5), .2)
        for scale in (.25, 2., 7.):
            with self.subTest(scale=scale):
                torch.testing.assert_close(score(air*scale, contact*scale, periods*scale, math.sqrt(.5), .2), expected)

    def test_each_comparison_uses_mean_of_its_two_periods(self):
        actual = self.evaluate([[.02, 0, 0, 0]], [[0]*4], [[.2, .6, .2, .6]])
        # Foot 0 compares with foot 1 (T=.4), foot 2 (T=.2), foot 3 (T=.4).
        expected = math.exp(-(.05**2 + .1**2 + .05**2) / math.sqrt(.5))
        torch.testing.assert_close(actual, tensor([expected]))

    def test_correct_diagonal_alternation_beats_synchronous_hopping(self):
        correct = self.evaluate([[.05, .05, 0, 0]], [[0, 0, .05, .05]], [[.2]*4])
        hopping = self.evaluate([[.05]*4], [[0]*4], [[.2]*4])
        torch.testing.assert_close(correct, tensor([1.]))
        torch.testing.assert_close(hopping, tensor([math.exp(-8*.2**2/math.sqrt(.5))]))
        self.assertLess(hopping.item(), correct.item())

    def test_mean_hopping_score_no_longer_improves_with_shorter_cycle(self):
        fractions = torch.linspace(.01, .5, 50, dtype=DTYPE).unsqueeze(1).repeat(1, 4)
        averages = []
        for period in (.2, .4, 1.):
            air = fractions * period
            value = score(air, torch.zeros_like(air), torch.full_like(air, period), math.sqrt(.5), .2)
            averages.append(value.mean().item())
        self.assertAlmostEqual(max(averages), min(averages), places=14)


class CompletedPeriodTests(unittest.TestCase):
    def assert_period(self, fixture, value):
        self.assertTrue(fixture.term.valid_periods.all().item())
        torch.testing.assert_close(fixture.term.periods, torch.full_like(fixture.term.periods, value), rtol=1e-10, atol=1e-10)

    def test_initial_standing_contact_and_first_real_landing_do_not_form_cycle(self):
        f = Fixture()
        torch.testing.assert_close(f.advance(True, 5), tensor([0.]))
        f.advance(False, 5)
        torch.testing.assert_close(f.advance(True), tensor([0.]))
        self.assertFalse(f.term.valid_periods.any().item())
        f.advance(True, 4)
        f.advance(False, 5)
        self.assertGreater(f.advance(True).item(), 0.)
        self.assert_period(f, .2)

    def test_completed_period_avoids_contact_sensor_endpoint_overcount(self):
        f = Fixture()
        f.prime()
        self.assert_period(f, .2)
        ids = [f.sensor.body_names.index(name) for name in FEET]
        summed = f.sensor.data.last_air_time[:, ids] + f.sensor.data.last_contact_time[:, ids]
        torch.testing.assert_close(summed, torch.full_like(summed, .21))

    def test_substep_landing_offsets_are_preserved(self):
        f = Fixture()
        f.advance(True, 5)
        f.advance(False, 5)
        f.physics_pattern([False, False, True, True])
        f.advance(True, 4)
        f.advance(False, 5)
        f.physics_pattern([False, True, True, True])
        self.assert_period(f, .195)

    def test_duplicate_call_cannot_consume_same_touchdown_twice(self):
        f = Fixture()
        f.advance(True, 5)
        f.advance(False, 5)
        f.advance(True, 5)
        f.advance(False, 5)
        expected = f.advance(True)
        snapshot = f.term.periods.clone()
        for _ in range(3):
            torch.testing.assert_close(f.evaluate(), expected)
            torch.testing.assert_close(f.term.periods, snapshot)
            self.assertTrue(f.term.valid_periods.all().item())

    def test_partial_reset_preserves_other_environments_in_same_step(self):
        f = Fixture(batch=2)
        f.prime()
        previous = f.evaluate()
        f.sensor.reset([0])
        f.term.reset([0])
        result = f.evaluate()
        self.assertEqual(result[0].item(), 0.)
        torch.testing.assert_close(result[1], previous[1])
        self.assertFalse(f.term.valid_periods[0].any().item())
        self.assertTrue(f.term.valid_periods[1].all().item())
        f.term.reset()
        self.assertFalse(f.term.valid_periods.any().item())

    def test_period_updates_after_frequency_change(self):
        f = Fixture()
        f.prime()
        self.assert_period(f, .2)
        f.advance(False, 10)
        f.advance(True, 10)
        f.advance(False, 10)
        f.advance(True, 1)
        self.assert_period(f, .4)

    def test_touchdown_at_exact_staleness_boundary_is_not_late_sampled_away(self):
        f = Fixture()
        f.prime()
        f.advance(False, 15)
        # The touchdown is exactly 2*old_period after the preceding event;
        # the control reward is sampled .02 s later, which is not staleness.
        f.advance(True)
        self.assert_period(f, .4)

    def test_stale_cycle_requires_two_fresh_touchdowns(self):
        f = Fixture()
        f.prime()
        self.assert_period(f, .2)
        torch.testing.assert_close(f.advance(True, 16), tensor([0.]))
        self.assertFalse(f.term.valid_periods.any().item())
        f.advance(False, 5)
        torch.testing.assert_close(f.advance(True, 5), tensor([0.]))
        f.advance(False, 5)
        self.assertGreater(f.advance(True).item(), 0.)
        self.assert_period(f, .2)

    def test_short_cycle_is_rejected_instead_of_clipped_or_using_old_period(self):
        f = Fixture()
        f.prime()
        for _ in range(3):
            f.advance(False)
            result = f.advance(True)
        self.assertFalse(f.term.valid_periods.any().item())
        torch.testing.assert_close(result, tensor([0.]))

    def test_subcontrol_air_or_support_phase_is_rejected(self):
        for pattern in ([False] + [True]*19, [False]*19 + [True]):
            with self.subTest(air_samples=sum(not x for x in pattern)):
                f = Fixture()
                f.advance(True, 5)
                for _ in range(3):
                    result = f.physics_pattern(pattern)
                # Total period .1 is legal; either the air or support phase
                # lasts .005, below a resolvable .02-second control phase.
                self.assertFalse(f.term.valid_periods.any().item())
                torch.testing.assert_close(result, tensor([0.]))

    def test_missed_short_contact_cannot_inflate_the_measured_period(self):
        f = Fixture()
        f.prime()
        f.advance(False, 5)
        # An entire .005-second contact occurs inside a control step, so the
        # public first-contact signal does not report it at the sample time.
        f.physics_pattern([True, False, False, False])
        f.advance(False, 4)
        result = f.advance(True)
        self.assertFalse(f.term.valid_periods.any().item())
        torch.testing.assert_close(result, tensor([0.]))

    def test_large_global_step_does_not_lose_period_precision(self):
        f = Fixture()
        for name, value in vars(f.sensor.data).items():
            setattr(f.sensor.data, name, value.float())
        f.term = Reward(NS(params=f.params), f.env)
        f.env.common_step_counter = 1_000_000_000
        f.prime()
        self.assertTrue(f.term.valid_periods.all().item())
        torch.testing.assert_close(f.term.periods, torch.full_like(f.term.periods, .2), rtol=1e-6, atol=1e-7)

    def test_float32_event_window_accepts_four_but_not_five_physics_steps(self):
        for timestamp in (8., 20.):
            with self.subTest(timestamp=timestamp):
                f = Fixture()
                for name, value in vars(f.sensor.data).items():
                    setattr(f.sensor.data, name, value.float())
                f.sensor.timestamp = torch.tensor([timestamp], dtype=torch.float32)
                for _ in range(4):
                    f.sensor.physics_sample(f.contacts(True), f.env.physics_dt)
                ids = [f.sensor.body_names.index(name) for name in FEET]
                self.assertTrue(f.sensor.compute_first_contact(.02, abs_tol=1e-5)[:, ids].all().item())
                if timestamp == 8.:
                    # Here four increments total .020000457763671875, so
                    # the sensor API's default tolerance misses the event.
                    self.assertFalse(f.sensor.compute_first_contact(.02)[:, ids].any().item())
                f.sensor.physics_sample(f.contacts(True), f.env.physics_dt)
                self.assertFalse(f.sensor.compute_first_contact(.02, abs_tol=1e-5)[:, ids].any().item())

    def test_float32_timestamp_boundary_accepts_minimum_air_and_support_phases(self):
        for timestamp in (8., 20.):
            for air_steps, contact_steps in ((1, 3), (3, 1)):
                with self.subTest(timestamp=timestamp, air_steps=air_steps, contact_steps=contact_steps):
                    f = Fixture()
                    for name, value in vars(f.sensor.data).items():
                        setattr(f.sensor.data, name, value.float())
                    f.sensor.timestamp = torch.tensor([timestamp], dtype=torch.float32)
                    f.term = Reward(NS(params=f.params), f.env)
                    f.advance(True, contact_steps)
                    f.advance(False, air_steps)
                    f.advance(True, contact_steps)
                    f.advance(False, air_steps)
                    result = f.advance(True)
                    # Around 20 s, four float32 timestamp increments total
                    # .019996643 s. A real four-substep phase must remain valid.
                    self.assertGreater(result.item(), 0.)
                    self.assertTrue(f.term.valid_periods.all().item())
                    torch.testing.assert_close(f.term.periods, torch.full_like(f.term.periods, .08), rtol=1e-6, atol=1e-7)

    def test_stop_clears_history_but_drift_and_turn_command_remain_active(self):
        f = Fixture(batch=4)
        f.prime()
        f.command[:] = tensor([[0., 0., 0.], [0., 0., 0.], [.1, 0., 0.], [0., 0., .2]])
        f.asset.data.root_com_lin_vel_b[:, 0] = tensor([0., .6, .5, 0.])
        result = f.advance(True)
        torch.testing.assert_close(result[[0, 2]], tensor([0., 0.]))
        self.assertFalse(f.term.valid_periods[[0, 2]].any().item())
        self.assertTrue(f.term.valid_periods[[1, 3]].all().item())
        self.assertTrue((result[[1, 3]] > 0).all().item())

    def test_upright_gate_and_sensor_body_reordering(self):
        f = Fixture(batch=3, body_names=["RR_foot", "trunk", "FR_foot", "FL_foot", "RL_foot"])
        f.prime()
        self.assert_period(f, .2)
        f.asset.data.projected_gravity_b[:, 2] = tensor([-.7, -.35, .1])
        result = f.evaluate()
        self.assertAlmostEqual(result[1].item(), .5 * result[0].item())
        self.assertEqual(result[2].item(), 0.)

    def test_no_reward_until_every_foot_has_a_complete_cycle(self):
        f = Fixture()
        f.advance(True, 5)
        for _ in range(2):
            f.advance([False, False, False, True], 5)
            result = f.advance(True, 5)
        self.assertTrue(f.term.valid_periods[0, :3].all().item())
        self.assertFalse(f.term.valid_periods[0, 3].item())
        torch.testing.assert_close(result, tensor([0.]))

    def test_invalid_configuration_rejected(self):
        for changes in (
            {"phase_error_scale": 0.}, {"max_phase_error": float("nan")},
            {"period_range": (.01, 2.)}, {"period_range": (2., .08)},
            {"stale_periods": .5}, {"synced_feet_pair_names": (("FR_foot", "FR_foot"), PAIRS[1])},
            {"synced_feet_pair_names": (("missing", "RL_foot"), PAIRS[1])},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Fixture(params=changes)
        f = Fixture()
        f.sensor.cfg.track_air_time = False
        with self.assertRaises(ValueError):
            Reward(NS(params=f.params), f.env)


class ConfigIntegrationTests(unittest.TestCase):
    def test_standard_walk_selects_normalized_reward_and_disables_airtime(self):
        tree = ast.parse(CONFIG.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TQBot2StandardWalkRoughEnvCfg")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
        prefixes = ("self.rewards.feet_gait.", "self.rewards.feet_air_time.", "gait_params")
        nodes = [n for n in init.body if ast.unparse(n).startswith(prefixes)]
        gait = NS(params={"sensor_cfg": NS(), "asset_cfg": NS(), "std": math.sqrt(.5), "max_err": .2})
        airtime = NS(params={"sensor_cfg": NS()})
        context = {
            "self": NS(rewards=NS(feet_gait=gait, feet_air_time=airtime), foot_link_name=".*_foot"),
            "PeriodNormalizedGaitReward": Reward, "math": math,
            "SceneEntityCfg": lambda name, **kwargs: NS(name=name, **kwargs),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONFIG), "exec"), context)
        self.assertIs(gait.func, Reward)
        self.assertEqual(gait.weight, .5)
        self.assertEqual(airtime.weight, 0.)
        self.assertNotIn("std", gait.params)
        self.assertNotIn("max_err", gait.params)
        self.assertEqual(gait.params["max_phase_error"], .2)
        self.assertEqual(set(gait.params["synced_feet_pair_names"]), set(PAIRS))


if __name__ == "__main__":
    unittest.main()
