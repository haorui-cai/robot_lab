"""Actual-revolution rewards: variable speed, reversals, resets and yaw gating."""

import importlib.util
import math
import unittest
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/cycle_tracking.py"
spec = importlib.util.spec_from_file_location("cycle_tracking_actual_tests", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ActualCycleTests(unittest.TestCase):
    def tracker(self, count=1, **kwargs):
        return module.CycleTracker(count, "cpu", actual_cycle_rewards=True, yaw_min_speed=6.0, **kwargs)

    def test_blend_equals_two_historical_policies_at_each_step(self):
        old = module.CycleTracker(3, "cpu")
        new = self.tracker(3, yaw_tracking_std=.125, yaw_success_tolerance=.06)
        blended = self.tracker(3, yaw_tracking_std=.125, yaw_success_tolerance=.06,
                               legacy_reward_fraction=.5)
        command = torch.tensor([[-10., 0.], [-14., .3], [-6., -.2]])
        for step in range(180):
            if step == 95:
                for state in (old, new, blended):
                    state.reset(torch.tensor([1]))
            roll = torch.tensor([-10. if step < 60 else -6.6,
                                 -8. if step < 110 else 4., -6.])
            yaw = torch.tensor([.1 if step % 2 else -.2, .35, -.2])
            for state in (old, new, blended):
                state.update(roll, yaw, command, .02)
            torch.testing.assert_close(blended.roll_payment, .5 * (old.roll_payment + new.roll_payment))
            torch.testing.assert_close(blended.yaw_payment, .5 * (old.yaw_payment + new.yaw_payment))
            torch.testing.assert_close(blended.yaw_good, new.yaw_good)
            torch.testing.assert_close(blended.window.mean, new.window.mean)
        # The exact -6 branch is closed in the newer policy, not in the older one.
        self.assertEqual(new.yaw_payment[2].item(), 0.)
        self.assertAlmostEqual(blended.yaw_payment[2].item(), .5, places=5)

    def test_blend_preserves_half_legacy_credit_before_first_lap(self):
        state = self.tracker(1, yaw_tracking_std=.125, legacy_reward_fraction=.5)
        state.update(torch.tensor([-10.]), torch.tensor([0.]), torch.tensor([[-10., 0.]]), .02)
        self.assertFalse(state.actual_window.valid.item())
        self.assertAlmostEqual(state.roll_payment.item(), .5)
        self.assertAlmostEqual(state.yaw_payment.item(), .5)

    def test_tight_yaw_reward_at_positive_and_negative_errors(self):
        state = self.tracker(4, yaw_tracking_std=.125, yaw_success_tolerance=.06)
        command = torch.tensor([[-10., 0.], [-10., 0.], [-10., .3], [-10., -.3]])
        errors = torch.tensor([.05, -.05, .05, -.05])
        for _ in range(100):
            state.update(torch.full((4,), -10.), command[:, 1] + errors, command, .02)
        torch.testing.assert_close(state.yaw_payment, torch.full((4,), math.exp(-.16)))

    def test_tight_yaw_success_inclusive_boundary(self):
        errors = torch.tensor([.0599, .06, .0601, -.0599, -.06, -.0601])
        state = self.tracker(6, yaw_tracking_std=.125, yaw_success_tolerance=.06)
        command = torch.tensor([[-10., 0.]]).repeat(6, 1)
        for _ in range(100):
            state.update(torch.full((6,), -10.), errors, command, .02)
        self.assertTrue((state.cycles > 0).all())
        torch.testing.assert_close(state.yaw_good, state.cycles * torch.tensor([1., 1., 0., 1., 1., 0.]))

    def test_actual_variable_speed_cycle_and_time_averaged_yaw(self):
        state = self.tracker()
        command = torch.tensor([[-10.0, 1 / 3]])
        for step in range(75):
            fast = step >= 50
            state.update(torch.tensor([-4 * math.pi if fast else -2 * math.pi]),
                         torch.tensor([1.0 if fast else 0.0]), command, 0.01)
        self.assertTrue(state.actual_window.valid.item())
        self.assertAlmostEqual(state.actual_window.duration.item(), 0.75, places=5)
        torch.testing.assert_close(state.actual_window.mean, torch.tensor([[-2 * math.pi / .75, 1 / 3]]))
        self.assertGreater(abs(state.window.mean[0, 0].item() - state.actual_window.mean[0, 0].item()), 0.1)
        self.assertAlmostEqual(state.yaw_payment.item(), 1.0, places=5)

    def test_no_tracking_credit_before_first_revolution(self):
        state = self.tracker()
        for _ in range(10):
            state.update(torch.tensor([-10.0]), torch.tensor([0.0]), torch.tensor([[-10.0, 0.0]]), .02)
        self.assertFalse(state.actual_window.valid.item())
        self.assertEqual(state.roll_payment.item(), 0)
        self.assertEqual(state.yaw_payment.item(), 0)

    def test_fractional_boundary_is_independent_of_command_period(self):
        state = self.tracker(2)
        for _ in range(100):
            state.update(torch.full((2,), -8.7), torch.full((2,), .2),
                         torch.tensor([[-6., .2], [-14., .2]]), .02)
        torch.testing.assert_close(state.actual_window.mean, torch.tensor([[-8.7, .2], [-8.7, .2]]))
        torch.testing.assert_close(state.actual_window.duration, torch.full((2,), 2 * math.pi / 8.7))
        torch.testing.assert_close(state.yaw_payment, torch.ones(2))
        self.assertNotAlmostEqual(state.roll_payment[0].item(), state.roll_payment[1].item())

    def test_stop_includes_dwell_and_does_not_reuse_fast_lap(self):
        state = self.tracker()
        command = torch.tensor([[-10., 0.]])
        for _ in range(100):
            state.update(torch.tensor([-10.]), torch.tensor([0.]), command, .02)
        self.assertEqual(state.yaw_payment.item(), 1)
        for _ in range(50):
            state.update(torch.tensor([0.]), torch.tensor([0.]), command, .02)
        expected = -2 * math.pi / (2 * math.pi / 10 + 1)
        self.assertAlmostEqual(state.actual_window.mean[0, 0].item(), expected, places=5)
        self.assertEqual(state.yaw_payment.item(), 0)

    def test_reverse_rocking_and_threshold_cannot_unlock_yaw(self):
        state = self.tracker(5)
        command = torch.tensor([[-14., .3]]).repeat(5, 1)
        for step in range(200):
            state.update(torch.tensor([6.1, -6.0, -6.1, 0., -10. if step % 2 else 10.]),
                         torch.full((5,), .3), command, .02)
        torch.testing.assert_close(state.yaw_payment, torch.tensor([0., 0., 1., 0., 0.]))
        self.assertLess(state.roll_payment[2].item(), 1e-10)

    def test_reversal_is_subtracted_and_can_invalidate_a_prior_cycle(self):
        state = self.tracker()
        command = torch.tensor([[-10., 0.]])
        for _ in range(100):
            state.update(torch.tensor([-7.]), torch.tensor([0.]), command, .01)
        self.assertTrue(state.actual_window.valid.item())
        for _ in range(100):
            state.update(torch.tensor([7.]), torch.tensor([0.]), command, .01)
        self.assertFalse(state.actual_window.valid.item())
        self.assertEqual(state.yaw_payment.item(), 0)

    def test_partial_reset_and_history_wrap(self):
        state = self.tracker(2, history_capacity=60)
        command = torch.tensor([[-10., .2]]).repeat(2, 1)
        for _ in range(200):
            state.update(torch.full((2,), -10.), torch.full((2,), .2), command, .02)
        before = state.actual_window.mean[1].clone()
        state.reset(torch.tensor([0]))
        self.assertFalse(state.actual_window.valid[0])
        torch.testing.assert_close(state.actual_window.mean[1], before)
        state.update(torch.full((2,), -10.), torch.full((2,), .2), command, .02)
        torch.testing.assert_close(state.yaw_payment, torch.tensor([0., 1.]))

    def test_existing_policy_observations_and_curriculum_counters_unchanged(self):
        legacy, updated = module.CycleTracker(1, "cpu"), self.tracker()
        command = torch.tensor([[-9., .1]])
        for step in range(100):
            roll, yaw = torch.tensor([-7. if step % 2 else -11.]), torch.tensor([.1])
            for state in (legacy, updated):
                state.update(roll, yaw, command, .02)
            torch.testing.assert_close(legacy.window.mean, updated.window.mean)
            for key in ("progress", "elapsed", "cycles", "roll_good", "yaw_good"):
                torch.testing.assert_close(getattr(legacy, key), getattr(updated, key))


if __name__ == "__main__":
    unittest.main()
