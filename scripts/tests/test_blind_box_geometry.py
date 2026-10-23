"""CPU checks that ground feet, hanging rear legs, and brief touches cannot pass."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

path = (Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/"
        "config/quadruped/tqbot2/blind_box_geometry.py")
spec = importlib.util.spec_from_file_location("blind_box_geometry", path)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)

warm_path = path.parent / "agents/blind_box_warm_start.py"
warm_spec = importlib.util.spec_from_file_location("blind_box_warm_start", warm_path)
warm = importlib.util.module_from_spec(warm_spec)
warm_spec.loader.exec_module(warm)


class BlindBoxTests(unittest.TestCase):
    def test_mesh_top_and_approach_ground(self):
        mesh, origins = geometry.build_box_mesh()
        self.assertTrue(mesh.is_watertight)
        vertices = np.asarray(mesh.vertices).reshape(-1, 2, 8, 3) - origins[:, None, None, :]
        np.testing.assert_allclose(vertices[:, 0, :, 2].max(axis=-1), 0, atol=1e-7)
        np.testing.assert_allclose(vertices[:, 1, :, 2].max(axis=-1),
                                   np.repeat(geometry.BOX_HEIGHTS, 3), atol=1e-7)
        np.testing.assert_allclose(vertices[:, 1, :, 0].min(axis=-1), geometry.BOX_FRONT, atol=1e-7)

    def test_ground_hanging_feet_and_edge_are_not_top_support(self):
        radius = torch.full((4,), 0.012)
        h, length = torch.tensor([0.04, 0.16]), torch.tensor([1.2, 1.6])
        feet = torch.tensor([[[1.4, -0.2, 0.052], [1.4, 0.2, 0.052],
                              [1.1, -0.2, 0.052], [1.1, 0.2, 0.052]]]).repeat(2, 1, 1)
        feet[1, :, 2] = 0.172
        self.assertTrue(geometry.feet_on_top(feet, h, length, radius).all())
        feet[:, 2:, 2] = 0.012
        self.assertFalse(geometry.feet_on_top(feet, h, length, radius).all(dim=1).any())
        feet[:, :, 2] = h[:, None] + radius
        feet[:, 0, 0] = 1.01
        self.assertFalse(geometry.feet_on_top(feet, h, length, radius).all(dim=1).any())

    def test_stability_rejects_airborne_and_moving_robot(self):
        feet = torch.tensor([[[1.4, -0.2, 0.112], [1.4, 0.2, 0.112],
                              [1.1, -0.2, 0.112], [1.1, 0.2, 0.112]]])
        args = [feet, torch.tensor([0.1]), torch.tensor([1.2]), torch.full((4,), 0.012),
                torch.tensor([[0., 0., -1.]]), torch.zeros(1, 3), torch.zeros(1, 3),
                torch.tensor([[1.25, 0., 0.4]]), torch.tensor([[[0., 0., 30.]] * 4])]
        self.assertTrue(geometry.stable_top(*args).item())
        args[-1].zero_()
        self.assertFalse(geometry.stable_top(*args).item())
        args[-1][:, :, 2] = 30
        args[5][:, 0] = 0.3
        self.assertFalse(geometry.stable_top(*args).item())

    def test_course_needs_two_frontier_batches(self):
        course = geometry.BoxCourse()
        self.assertFalse(course.observe(127, 127))
        self.assertFalse(course.observe(1, 1))
        self.assertEqual(course.level, 0)
        self.assertFalse(course.observe(128, 0))
        self.assertFalse(course.observe(128, 128))
        self.assertTrue(course.observe(128, 128))
        self.assertEqual(course.level, 1)
        for _ in range(50):
            course.observe(128, 128)
        self.assertEqual(course.level, len(geometry.BOX_HEIGHTS) - 1)

    def test_landing_allows_walking_but_requires_each_foot_contact(self):
        feet = torch.tensor([[[1.7, -0.2, 0.15], [1.7, 0.2, 0.112],
                              [1.2, -0.2, 0.112], [1.2, 0.2, 0.15]]])
        ages = torch.tensor([[0.1, 0.0, 0.0, 0.1]])
        args = [feet, torch.tensor([0.1]), torch.tensor([1.2]), torch.tensor([[0., 0., -1.]]),
                torch.tensor([[0.4, 0., 0.]]), torch.zeros(1, 3), torch.tensor([[1.5, 0., 0.4]]), ages]
        self.assertTrue(geometry.stable_landing(*args).item())
        ages[0, 3] = 1e6  # a rear foot that never landed must not qualify
        self.assertFalse(geometry.stable_landing(*args).item())
        ages[0, 3] = 0.1
        args[4][0, 2] = 0.5  # jumping past the platform is not a stable landing
        self.assertFalse(geometry.stable_landing(*args).item())

    def test_walking_actor_migration_preserves_actions(self):
        torch.manual_seed(5)
        old = {"mlp.0.weight": torch.randn(512, 45), "mlp.0.bias": torch.randn(512)}
        target = {"mlp.0.weight": torch.empty(512, 684), "mlp.0.bias": torch.empty(512)}
        lifted = warm.lift_walking_actor(old, target)
        history = torch.randn(32, 684)
        original = torch.nn.functional.linear(history[:, warm.latest_walking_columns()],
                                              old["mlp.0.weight"], old["mlp.0.bias"])
        expanded = torch.nn.functional.linear(history, lifted["mlp.0.weight"], lifted["mlp.0.bias"])
        torch.testing.assert_close(original, expanded, rtol=1e-5, atol=1e-5)
        with self.assertRaises(ValueError):
            warm.lift_walking_actor({**old, "obs_normalizer._mean": torch.zeros(45)}, target)

    def test_history_training_preserves_prior_and_can_learn_new_columns(self):
        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.obs_normalizer = torch.nn.Identity()
                self.mlp = torch.nn.Sequential(torch.nn.Linear(684, 16), torch.nn.ELU(), torch.nn.Linear(16, 12))

        actor = Actor()
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        original = {key: value.clone() for key, value in actor.state_dict().items()}
        warm.train_history_adapter_only(actor, optimizer)
        actor.mlp(torch.randn(8, 684)).square().sum().backward()
        optimizer.step()
        columns = warm.latest_walking_columns()
        self.assertTrue(torch.equal(actor.mlp[0].weight[:, columns], original["mlp.0.weight"][:, columns]))
        self.assertTrue(torch.equal(actor.mlp[0].bias, original["mlp.0.bias"]))
        self.assertTrue(torch.equal(actor.mlp[2].weight, original["mlp.2.weight"]))
        self.assertFalse(torch.equal(actor.mlp[0].weight, original["mlp.0.weight"]))


if __name__ == "__main__":
    unittest.main()
