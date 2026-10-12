"""Regression checks for planted, dragged and swinging feet on flat/slope terrain."""

import importlib.util
from pathlib import Path
import unittest

import torch

path = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/foot_motion.py"
spec = importlib.util.spec_from_file_location("foot_motion", path)
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


class FootMotionTests(unittest.TestCase):
    def test_rotating_calf_requires_point_velocity_not_com_velocity(self):
        pos = torch.zeros(1, 1, 3)
        quat = torch.tensor([[[2.**-.5, 0., 2.**-.5, 0.]]])
        point, vel = motion.foot_point_state(pos, quat, pos, torch.tensor([[[0., 1., 0.]]]), (0., 0., -.153))
        torch.testing.assert_close(point, torch.tensor([[[-.153, 0., 0.]]]), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(vel, torch.tensor([[[0., 0., .153]]]), atol=1e-7, rtol=1e-6)

    def test_planted_foot_is_free_but_body_following_drag_is_penalized(self):
        foot = torch.tensor([[[0., 0., 0.]], [[.5, 0., 0.]]])
        base = torch.tensor([[.5, 0., 0.], [.5, 0., 0.]])
        normal = torch.tensor([[[0., 0., 1.]], [[0., 0., 1.]]])
        slip, swing = motion.foot_motion_costs(torch.zeros(2, 1), foot, base, normal,
                                             torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1, dtype=torch.bool))
        torch.testing.assert_close(slip, torch.tensor([0., .5]))
        torch.testing.assert_close(swing, torch.zeros(2))

    def test_moving_swing_needs_clearance_but_no_reward_for_excess_lift(self):
        clearance = torch.tensor([[0.], [.015], [.03], [.08]])
        foot = torch.tensor([[[.4, 0., 0.]]]).repeat(4, 1, 1)
        normal = torch.tensor([[[0., 0., 1.]]]).repeat(4, 1, 1)
        slip, swing = motion.foot_motion_costs(clearance, foot, torch.zeros(4, 3), normal,
                                             torch.zeros(4, 1, dtype=torch.bool), torch.ones(4, 1, dtype=torch.bool))
        torch.testing.assert_close(slip, torch.zeros(4))
        torch.testing.assert_close(swing, torch.tensor([1., .25, 0., 0.]))

    def test_slope_normal_motion_is_not_tangential_slip(self):
        normal = torch.tensor([[[0., -.5, 3.**.5 / 2]]])
        tangent = torch.tensor([[[0., 3.**.5 / 2, .5]]])
        torch.testing.assert_close(motion.tangent_velocity(normal * .2, normal), torch.zeros_like(normal), atol=1e-7, rtol=0.)
        torch.testing.assert_close(motion.tangent_velocity(tangent, normal), tangent)

    def test_missing_ray_and_leg_contact_above_ground_do_not_become_stance(self):
        clearance = torch.tensor([[.08], [0.]])
        foot = torch.tensor([[[.4, 0., 0.]]]).repeat(2, 1, 1)
        normal = torch.tensor([[[0., 0., 1.]]]).repeat(2, 1, 1)
        slip, swing = motion.foot_motion_costs(clearance, foot, torch.zeros(2, 3), normal,
                                             torch.ones(2, 1, dtype=torch.bool), torch.tensor([[True], [False]]))
        torch.testing.assert_close(slip, torch.zeros(2))
        torch.testing.assert_close(swing, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
