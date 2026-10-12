"""CPU checks for recovery gating and per-episode torque history."""

import importlib.util
from pathlib import Path
import unittest

import torch

MODULE = Path(__file__).resolve().parents[2] / (
    "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/recovery_costs.py"
)
spec = importlib.util.spec_from_file_location("recovery_costs", MODULE)
costs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(costs)


class RecoveryCostTests(unittest.TestCase):
    def test_gate_distinguishes_walking_tilt_and_grounded_trunk(self):
        gate = costs.recovery_gate(torch.tensor([1.0, 0.93, 0.675, -1.0, 1.0]),
                                   torch.tensor([0., 0., 0., 0., 25.]))
        torch.testing.assert_close(gate, torch.tensor([0., 0., .5, 1., 1.]))
        torch.testing.assert_close(costs.recovery_gate(torch.tensor([3.0**0.5 / 2]), torch.zeros(1)), torch.zeros(1))

    def test_effort_and_speed_are_sign_symmetric_and_normalized(self):
        torque = torch.tensor([[23.7, -35.55], [-23.7, 35.55]])
        limits = torch.tensor([[23.7, 35.55], [23.7, 35.55]])
        torch.testing.assert_close(costs.normalized_effort(torque, limits), torch.ones(2))
        torch.testing.assert_close(costs.normalized_velocity(torch.tensor([[8., -8.]]), 8.), torch.ones(1))

    def test_partial_reset_has_no_false_torque_jump_and_preserves_other_history(self):
        torque = torch.zeros(2, 3)
        limits = torch.ones_like(torque)
        history = costs.TorqueChangeHistory(torque)
        torch.testing.assert_close(history.compute(torch.ones_like(torque), limits, .02), torch.zeros(2))
        history.reset(torch.tensor([0]))
        torch.testing.assert_close(history.compute(torch.full_like(torque, 2.), limits, .02),
                                   torch.tensor([0., 1.]))
        torch.testing.assert_close(history.compute(torch.full_like(torque, 3.), limits, .01),
                                   torch.full((2,), 4.))
        history.reset()
        torch.testing.assert_close(history.compute(torch.zeros_like(torque), limits, .02), torch.zeros(2))

    def test_history_copies_values_instead_of_aliasing_live_torque_buffer(self):
        torque = torch.zeros(1, 2)
        history = costs.TorqueChangeHistory(torque)
        history.compute(torque, torch.ones_like(torque), .02)
        torque[:] = 1.
        torch.testing.assert_close(history.compute(torque, torch.ones_like(torque), .02), torch.ones(1))

    def test_energy_reward_requires_motion_command_and_decreases_with_work(self):
        power = torch.tensor([0., 50., 100., 100.])
        speed = torch.tensor([0., .5, .5, .5])
        tracking = torch.tensor([1., 1., 1., .2])
        score = costs.distance_energy_score(power, speed, tracking, torch.tensor([False, True, True, True]))
        self.assertTrue(torch.isfinite(score).all())
        self.assertEqual(score[0].item(), 0.)
        self.assertGreater(score[1].item(), score[2].item())
        self.assertGreater(score[2].item(), score[3].item())


if __name__ == "__main__":
    unittest.main()
