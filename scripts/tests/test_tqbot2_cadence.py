"""CPU behavioral tests for measured full-cycle cadence; no simulator startup."""

import ast
import math
from types import SimpleNamespace as NS
import unittest

import torch

from test_tqbot2_normalized_gait import (
    DTYPE, FEET, SOURCE, Fixture as GaitFixture, ManagerTermBase, tensor,
)


def production_symbols():
    tree = ast.parse(SOURCE.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *definitions,
    ], type_ignores=[])
    namespace = {"torch": torch, "math": math, "ManagerTermBase": ManagerTermBase}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["cadence_score"], namespace["FootCadenceReward"]


score, Reward = production_symbols()


class Fixture(GaitFixture):
    def __init__(self, batch=1, body_names=None, params=None):
        super().__init__(batch=batch, body_names=body_names)
        self.params = dict(
            target_frequency=2.0, frequency_scale=.5,
            command_name="base_velocity", command_threshold=.1,
            asset_cfg=NS(name="robot"),
            sensor_cfg=NS(name="contact_forces", body_ids=[
                self.sensor.body_names.index(name) for name in FEET
            ]),
            period_range=(.08, 2.), stale_periods=2.,
        )
        self.params.update(params or {})
        self.term = Reward(NS(params=self.params), self.env)

    def prime_target(self):
        # 0.30 s stance + 0.20 s swing = a real completed 0.50 s cycle.
        self.advance(True, 15)
        self.advance(False, 10)
        self.advance(True, 15)
        self.advance(False, 10)
        return self.advance(True)

    def sample_periodic(self, period, seconds=5., duty=.6):
        """Replay physics contacts and collect one score per control step."""
        physics_samples = round(seconds / self.env.physics_dt)
        period_samples = round(period / self.env.physics_dt)
        stance_samples = round(period_samples * duty)
        values = []
        for index in range(physics_samples):
            self.sensor.physics_sample(self.contacts(index % period_samples < stance_samples), self.env.physics_dt)
            if index % 4 == 3:
                self.env.common_step_counter += 1
                values.append(self.evaluate().clone())
        return torch.stack(values)


class CadenceKernelTests(unittest.TestCase):
    def test_target_is_unique_optimum_and_both_sides_lose_reward(self):
        frequencies = tensor([1., 1.5, 2., 2.5, 3., 4., 4.5])
        periods = frequencies.reciprocal().unsqueeze(1).repeat(1, 4)
        actual = score(periods, torch.zeros_like(periods), 2., .5)
        expected = (1. + ((frequencies - 2.) / .5).square()).reciprocal()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.argmax().item(), 2)
        self.assertEqual(actual[2].item(), 1.)
        self.assertTrue((actual[3:] < actual[2:-1]).all().item())
        self.assertGreater(actual[-1].item(), .03)  # useful shaping at the current ~4.5 Hz

    def test_bad_foot_loses_its_share_and_invalidating_it_cannot_improve_score(self):
        periods = tensor([[.5, .5, .5, .25]])
        actual = score(periods, torch.zeros_like(periods), 2., .5)
        torch.testing.assert_close(actual, tensor([(3. + 1./17.) / 4.]))
        valid = torch.tensor([[True, True, True, False]])
        missing = score(periods, torch.zeros_like(periods), 2., .5, valid)
        torch.testing.assert_close(missing, tensor([.75]))
        self.assertLess(missing.item(), actual.item())
        self.assertLess(actual.item(), 1.)

    def test_zero_to_four_valid_feet_use_fixed_total_foot_count(self):
        periods = torch.full((5, 4), .5, dtype=DTYPE)
        valid = torch.arange(4).unsqueeze(0) < torch.arange(5).unsqueeze(1)
        actual = score(periods, torch.zeros_like(periods), 2., .5, valid)
        torch.testing.assert_close(actual, tensor([0., .25, .5, .75, 1.]))

    def test_invalid_history_does_not_contaminate_other_foot_scores(self):
        periods = tensor([[.5, float("nan"), 0., float("inf")]])
        valid = torch.tensor([[True, False, False, False]])
        actual = score(periods, torch.zeros_like(periods), 2., .5, valid)
        torch.testing.assert_close(actual, tensor([.25]))

    def test_open_overdue_cycle_decays_and_never_improves_prior_fast_score(self):
        elapsed = tensor([.02, .25, .5, .6, .75, 1.]).unsqueeze(1).repeat(1, 4)
        target = score(torch.full_like(elapsed, .5), elapsed, 2., .5)
        torch.testing.assert_close(target[:3], tensor([1., 1., 1.]))
        self.assertTrue((target[3:] < target[2:-1]).all().item())
        for old_period in (.25, .4, .75):
            values = score(torch.full_like(elapsed, old_period), elapsed, 2., .5)
            self.assertTrue((values[1:] <= values[:-1]).all().item())


class CadenceEventTests(unittest.TestCase):
    def test_true_touchdown_period_at_target_has_unit_reward(self):
        f = Fixture()
        torch.testing.assert_close(f.prime_target(), tensor([1.]))
        torch.testing.assert_close(f.term.periods, tensor([[.5] * 4]))
        # ContactSensor last phase timers each include their transition sample;
        # using their naive sum would erroneously measure 0.51 s instead.
        ids = f.term.foot_ids
        torch.testing.assert_close(
            f.sensor.data.last_air_time[:, ids] + f.sensor.data.last_contact_time[:, ids],
            tensor([[.51] * 4]),
        )

    def test_faster_touchdowns_do_not_create_more_reward_per_second(self):
        slow, fast = Fixture(), Fixture()
        slow_scores = slow.sample_periodic(.5)[100:]  # equal 3 s after warmup
        fast_scores = fast.sample_periodic(.25)[100:]
        torch.testing.assert_close(slow_scores, torch.ones_like(slow_scores))
        torch.testing.assert_close(fast_scores, torch.full_like(fast_scores, 1./17.))
        self.assertGreater(slow_scores.sum().item(), fast_scores.sum().item())

    def test_duty_factor_does_not_change_score_for_same_full_cycle(self):
        for duty in (.2, .4, .8):
            with self.subTest(duty=duty):
                f = Fixture()
                actual = f.sample_periodic(.5, duty=duty)[100:]
                torch.testing.assert_close(actual, torch.ones_like(actual))

    def test_standing_start_and_one_real_landing_do_not_earn_cadence(self):
        f = Fixture()
        self.assertEqual(f.advance(True, 100).item(), 0.)
        f.advance(False, 10)
        self.assertEqual(f.advance(True).item(), 0.)
        self.assertFalse(f.term.valid_periods.any().item())

    def test_stopping_command_clears_history_even_while_drifting(self):
        f = Fixture()
        f.prime_target()
        f.command.zero_()
        f.asset.data.root_com_lin_vel_b[:, 0] = 2.
        self.assertEqual(f.advance(True).item(), 0.)
        self.assertFalse(f.term.valid_periods.any().item())
        f.command[:, 0] = .5
        self.assertEqual(f.advance(True).item(), 0.)

    def test_standing_after_good_cycle_decays_then_expires(self):
        for contact in (True, False):
            with self.subTest(contact=contact):
                f = Fixture()
                f.prime_target()
                values = [f.advance(contact).item() for _ in range(55)]
                self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
                self.assertLess(values[29], .8)
                self.assertEqual(values[-1], 0.)
                self.assertFalse(f.term.valid_periods.any().item())

    def test_target_change_requires_new_completed_cycle_not_just_waiting(self):
        f = Fixture()
        f.sample_periodic(.25, seconds=2.)
        previous = f.evaluate().item()
        # Until a new landing exists, waiting must not transform 4 Hz to 2 Hz.
        for _ in range(20):
            self.assertLessEqual(f.advance(False).item(), previous + 1.e-12)
        # Two genuinely measured target cycles reacquire the target score.
        f.prime_target()
        torch.testing.assert_close(f.term.periods, tensor([[.5]*4]))
        self.assertAlmostEqual(f.evaluate().item(), 1.)

    def test_independent_completed_cycles_each_earn_one_quarter(self):
        for valid_count in (1, 2, 3):
            with self.subTest(valid_count=valid_count):
                f = Fixture()
                f.advance(True, 15)
                for _ in range(2):
                    f.advance([False] * valid_count + [True] * (4 - valid_count), 10)
                    result = f.advance(True, 15)
                self.assertTrue(f.term.valid_periods[0, :valid_count].all().item())
                self.assertFalse(f.term.valid_periods[0, valid_count:].any().item())
                self.assertAlmostEqual(result.item(), valid_count / 4.)

    def test_one_invalid_cycle_removes_only_that_foots_share(self):
        f = Fixture()
        f.prime_target()
        f.advance([False, True, True, True])
        result = f.advance(True)
        self.assertFalse(f.term.valid_periods[0, 0].item())
        self.assertTrue(f.term.valid_periods[0, 1:].all().item())
        torch.testing.assert_close(result, tensor([.75]))

    def test_one_stale_foot_does_not_erase_other_ongoing_cycles(self):
        f = Fixture()
        f.prime_target()
        for step in range(75):
            moving_feet_contact = step % 25 < 15
            result = f.advance([moving_feet_contact] * 3 + [True])
        self.assertTrue(f.term.valid_periods[0, :3].all().item())
        self.assertFalse(f.term.valid_periods[0, 3].item())
        torch.testing.assert_close(result, tensor([.75]))

    def test_selected_sensor_order_and_partial_reset_are_preserved(self):
        f = Fixture(batch=2, body_names=["RR_foot", "trunk", "FR_foot", "FL_foot", "RL_foot"])
        f.prime_target()
        self.assertEqual([f.sensor.body_names[i] for i in f.term.foot_ids], list(FEET))
        f.term.reset([0])
        f.sensor.reset([0])
        torch.testing.assert_close(f.evaluate(), tensor([0., 1.]))

    def test_duplicate_call_and_non_upright_gate(self):
        f = Fixture(batch=3)
        f.prime_target()
        f.asset.data.projected_gravity_b[:, 2] = tensor([-.7, -.35, .1])
        for _ in range(3):
            torch.testing.assert_close(f.evaluate(), tensor([1., .5, 0.]))

    def test_invalid_cycle_replaces_old_reward_with_zero(self):
        f = Fixture()
        f.prime_target()
        f.advance(False)
        self.assertEqual(f.advance(True).item(), 0.)
        self.assertFalse(f.term.valid_periods.any().item())

    def test_configuration_rejects_unreachable_or_invalid_targets(self):
        for changes in (
            {"target_frequency": 0.}, {"target_frequency": float("nan")},
            {"frequency_scale": 0.}, {"frequency_scale": float("inf")},
            {"target_frequency": .2}, {"target_frequency": 20.},
            {"stale_periods": float("nan")},
            {"sensor_cfg": NS(name="contact_forces", body_ids=[])},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Fixture(params=changes)


if __name__ == "__main__":
    unittest.main()
