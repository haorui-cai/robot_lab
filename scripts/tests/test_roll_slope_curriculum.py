"""CPU checks for slope geometry and independent difficulty gates."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/slope_tracking.py"
spec = importlib.util.spec_from_file_location("slope_tracking", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SlopeTests(unittest.TestCase):
    def test_success_gates_and_directions_are_independent(self):
        course = module.SlopeCourse()
        for step in (500, 1000, 1500):
            self.assertFalse(course.observe("uphill", 512, 0, 0, step))
        self.assertEqual(course.maximum("uphill"), 0)
        self.assertFalse(course.observe("downhill", 512, 480, 450, 500))
        self.assertTrue(course.observe("downhill", 512, 480, 450, 1000))
        self.assertEqual(course.maximum("downhill"), 3)
        self.assertEqual(course.maximum("uphill"), 0)
        # A failed batch interrupts the success streak.
        self.assertFalse(course.observe("uphill", 512, 480, 450, 2000))
        self.assertFalse(course.observe("uphill", 512, 0, 0, 2500))
        self.assertFalse(course.observe("uphill", 512, 480, 450, 3000))
        self.assertEqual(course.maximum("uphill"), 0)
        self.assertTrue(course.observe("uphill", 512, 480, 450, 3500))

    def test_final_ranges_and_batch_minimum(self):
        course = module.SlopeCourse()
        self.assertFalse(course.observe("uphill", 511, 511, 511, 500))
        self.assertEqual(course.batches["uphill"]["streak"], 0)
        for name, end in (("uphill", 10), ("downhill", 30)):
            for step in range(1000, 15000, 500):
                course.observe(name, 512, 512, 512, step)
            self.assertEqual(course.maximum(name), end)

    def test_real_mesh_slopes_have_correct_sign_and_normals(self):
        mesh, origins, angles = module.build_slope_mesh()
        self.assertTrue(mesh.is_watertight)
        self.assertEqual(angles.tolist(), list(range(-30, 11)))
        vertices = np.asarray(mesh.vertices).reshape(41, 8, 3) - origins[:, None, :]
        radians = np.deg2rad(angles)
        normals = np.stack((np.zeros(41), -np.sin(radians), np.cos(radians)), axis=-1)
        distance = np.sum(vertices * normals[:, None, :], axis=-1)
        np.testing.assert_allclose(distance.max(axis=-1), 0, atol=2e-6)
        np.testing.assert_allclose(distance.min(axis=-1), -0.2 * np.cos(radians), atol=2e-6)
        # The positive-y edge climbs uphill and descends downhill.
        high_y = vertices[:, :, 1] > 0
        edge_z = np.array([v[mask, 2].max() for v, mask in zip(vertices, high_y)])
        np.testing.assert_allclose(edge_z, 30 * np.tan(radians), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
