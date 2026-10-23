"""CPU tests of full-cycle credit and performance-gated progression."""

import importlib.util
import math
import unittest
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/cycle_tracking.py"
spec = importlib.util.spec_from_file_location("cycle_tracking", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CycleTracker, RollingCourse = module.CycleTracker, module.RollingCourse


class CycleTests(unittest.TestCase):
    def test_burst_and_coast_equal_uniform_cycle_average(self):
        state = CycleTracker(2, "cpu")
        target = torch.tensor([[-2 * math.pi, 0.2]]).repeat(2, 1)
        # One second, one revolution: one robot uniform, one alternates kick/coast.
        for step in range(200):
            rate = torch.tensor([-2 * math.pi, -4 * math.pi if step % 2 == 0 else 0.0])
            state.update(rate, torch.full((2,), 0.2), target, 0.01)
            if step >= 100:
                torch.testing.assert_close(state.roll_payment, torch.ones(2), atol=1e-4, rtol=0)
        torch.testing.assert_close(state.last_yaw, torch.full((2,), 0.2), atol=1e-5, rtol=0)

    def test_reverse_and_rocking_cannot_earn_cycles(self):
        state = CycleTracker(2, "cpu")
        command = torch.tensor([[-7.14, 0.0]]).repeat(2, 1)
        for step in range(200):
            state.update(torch.tensor([7.14, -7.14 if step % 2 else 7.14]), torch.zeros(2), command, 0.02)
            self.assertLess(state.roll_payment.sum().item(), 1e-8)
            self.assertLess(state.yaw_payment.sum().item(), 1e-8)
        self.assertEqual(state.cycles.sum().item(), 0)

    def test_partial_reset_preserves_other_environment(self):
        state = CycleTracker(2, "cpu")
        command = torch.tensor([[-7.14, 0.0]]).repeat(2, 1)
        for _ in range(50):
            state.update(torch.full((2,), -7.14), torch.zeros(2), command, 0.02)
        before = {key: value[1].item() for key, value in vars(state).items() if isinstance(value, torch.Tensor)}
        state.reset(torch.tensor([0]))
        for key, value in vars(state).items():
            if not isinstance(value, torch.Tensor):
                continue
            self.assertEqual(value[0].item(), 0)
            self.assertEqual(value[1].item(), before[key])
        state.update(torch.zeros(2), torch.zeros(2), command, 0.02)
        self.assertLess(state.roll_payment[0].item(), 1e-8)
        self.assertGreater(state.roll_payment[1].item(), 0.9)

    def test_fractional_period_and_partial_history(self):
        window = module.CycleWindow(2, "cpu")
        period = torch.tensor([0.815, 1.003])
        for step in range(150):
            window.update(torch.tensor([[-7.0, 0.3], [-8.0, -0.2]]), period, 0.02)
            torch.testing.assert_close(window.mean, torch.tensor([[-7.0, 0.3], [-8.0, -0.2]]), atol=1e-4, rtol=0)

    def test_curriculum_requires_success_not_time(self):
        course = RollingCourse()
        for step in (500, 1000, 1500):
            self.assertFalse(course.observe(2048, 0, 0, step))
        self.assertEqual(course.phase_level, 0)
        self.assertFalse(course.observe(2048, 1900, 1800, 2000))
        self.assertTrue(course.observe(2048, 1900, 1800, 2500))
        self.assertEqual((course.phase_level, course.yaw_level), (1, 0))
        self.assertFalse(course.observe(2048, 1900, 1800, 3000))
        self.assertTrue(course.observe(2048, 1900, 1800, 3500))
        self.assertEqual((course.phase_level, course.yaw_level), (1, 1))
        self.assertEqual(course.yaw_widths[-1], 0.5)

    def test_assistance_reduces_only_on_success_and_reaches_zero(self):
        course = RollingCourse((1.0, 0.8, 0.6, 0.4, 0.2, 0.0))
        step = 0
        # Repeated failure must preserve the initial assistance.
        for _ in range(3):
            step += 500
            self.assertFalse(course.observe(2048, 0, 0, step))
        self.assertEqual(course.initial_speed_scale, 1.0)
        expected_axes = ["assist", "phase", "yaw"] * 5 + ["phase", "yaw", "phase"]
        for axis in expected_axes:
            before = (course.assist_level, course.phase_level, course.yaw_level)
            step += 500
            self.assertFalse(course.observe(2048, 1900, 1800, step))
            step += 500
            self.assertTrue(course.observe(2048, 1900, 1800, step))
            after = (course.assist_level, course.phase_level, course.yaw_level)
            self.assertEqual(sum(a - b for a, b in zip(after, before)), 1)
            index = ("assist", "phase", "yaw").index(axis)
            self.assertEqual(after[index], before[index] + 1)
        self.assertEqual(course.initial_speed_scale, 0.0)
        self.assertEqual(course.phase_widths[course.phase_level], math.pi)
        self.assertEqual(course.yaw_widths[course.yaw_level], 0.5)
        for _ in range(4):
            step += 500
            self.assertFalse(course.observe(2048, 1900, 1800, step))

    def test_acceleration_and_partial_reset_do_not_cross_episodes(self):
        state = CycleTracker(2, "cpu")
        command = torch.tensor([[-10.0, 0.0]]).repeat(2, 1)
        zero = torch.zeros(2)
        state.reset(torch.tensor([0, 1]), initial_roll_rate=-7.0)
        state.update(torch.tensor([-9.0, -7.0]), zero, command, 0.02)
        torch.testing.assert_close(state.roll_acceleration, torch.tensor([-100.0, 0.0]))
        # Reset only env 0 from -9 to rest: the teleport is not acceleration.
        state.reset(torch.tensor([0]), initial_roll_rate=0.0)
        state.update(torch.tensor([0.0, -8.0]), zero, command, 0.02)
        torch.testing.assert_close(state.roll_acceleration, torch.tensor([0.0, -50.0]))
        state.update(torch.tensor([-2.0, -8.0]), zero, command, 0.02)
        torch.testing.assert_close(state.roll_acceleration, torch.tensor([-100.0, 0.0]))

    def test_yaw_acceleration_reset_and_first_sample(self):
        state = CycleTracker(2, "cpu")
        command = torch.tensor([[-10.0, 0.0]]).repeat(2, 1)
        roll = torch.full((2,), -10.0)
        # Without a previous sample, do not invent an acceleration.
        state.update(roll, torch.tensor([1.0, -1.0]), command, 0.02)
        torch.testing.assert_close(state.yaw_acceleration, torch.zeros(2))
        state.update(roll, torch.tensor([1.2, -1.4]), command, 0.02)
        torch.testing.assert_close(state.yaw_acceleration, torch.tensor([10.0, -20.0]))
        state.reset(torch.tensor([0]), initial_roll_rate=-10.0, initial_yaw_rate=0.0)
        state.update(roll, torch.tensor([0.0, -1.2]), command, 0.02)
        torch.testing.assert_close(state.yaw_acceleration, torch.tensor([0.0, 10.0]))
        state.update(roll, torch.tensor([0.2, -1.2]), command, 0.02)
        torch.testing.assert_close(state.yaw_acceleration, torch.tensor([10.0, 0.0]))

    def test_yaw_reward_is_gated_by_roll_tracking(self):
        state = CycleTracker(2, "cpu")
        command = torch.tensor([[-6.0, 0.3]]).repeat(2, 1)
        # Both track yaw perfectly; only one rolls at the commanded rate.
        for _ in range(60):
            state.update(torch.tensor([0.0, -6.0]), torch.full((2,), 0.3), command, 0.02)
        self.assertLess((5 * state.yaw_payment[0]).item(), 1e-6)
        self.assertAlmostEqual((5 * state.yaw_payment[1]).item(), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
